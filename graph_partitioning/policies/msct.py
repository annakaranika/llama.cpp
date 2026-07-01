"""Baechi memory-constrained SCT (m-SCT) baseline for single-request inference.

Faithful (operator-graph) implementation of the SCT / m-SCT algorithm:

  * Classic SCT (Hanen & Munier): under the "small communication time"
    assumption, an LP picks, for each operator, a single *favorite child* —
    the one successor worth co-locating to kill its communication arc — with
    each op having at most one favorite successor and one favorite predecessor
    (so favorites form co-location chains).
  * m-SCT (Baechi, SoCC 2020): the memory-constrained variant — the same
    favorite-child preference, but a list scheduler places operators on devices
    subject to per-device DRAM capacity, falling back to earliest-finish-time
    (ETF) when the favorite's device is full.

This operates on the transformer *operator DAG* (per layer: one op per KV head
of attention and one per FFN shard, joined by all-reduce nodes), not on whole
layers — so unlike ``hybrid_pp_tp`` it makes genuine per-operator placement
decisions driven by the graph's communication edges. Contrast with ``alpa``
(per-op tensor sharding via ILP) and ``metis`` (balanced k-way edge-cut).
"""

import time
from typing import Dict, List, Optional, Tuple

import pulp

from ..common import AttentionSplitStrategy, CommunicationModel, Device, FFNSplitStrategy
from ..graph_plots import draw_device_computation_breakdown, draw_partitions
from ..logging_utils import setup_logger, setup_output_dir
from ..model import ModelSpec
from ..network import Network
from .alpa import _SOLVER  # reuse the HiGHS/CBC solver picked once at import

LOGGER = setup_logger(name=__name__)


# ---------------------------------------------------------------------------
# Operator DAG
# ---------------------------------------------------------------------------
def _build_op_dag(
    model: ModelSpec, net: Network, seq_len: int, mem_seq_len: int
) -> Tuple[List[Dict], List[Tuple[int, int, float]]]:
    """Build the transformer operator DAG in topological order.

    Each node is a dict with: ``layer``, ``kind`` (attn|ffn|attn_ar|ffn_ar),
    ``split`` (head/shard index, -1 for AR), ``flops``, ``mem_gb``. Edges are
    ``(u, v, bytes)`` activation transfers (zero comm if u, v co-located).
    """
    num_kv = model.num_kv_heads
    num_ffn = model.get_max_ffn_splits(FFNSplitStrategy.KV_HEADS, len(net.servers))
    attn_flops = model.attn_compute_flops(seq_len)
    ffn_flops = model.ffn_compute_flops(seq_len)
    attn_mem = model.attn_memory_gb_per_kv_head(mem_seq_len)
    ffn_mem = model.ffn_memory_gb_per_fraction(1.0 / num_ffn)
    act_bytes = float(model.activation_bytes) * seq_len

    nodes: List[Dict] = []
    edges: List[Tuple[int, int, float]] = []

    def add(layer: int, kind: str, split: int, flops: float, mem: float) -> int:
        nodes.append({"layer": layer, "kind": kind, "split": split,
                      "flops": flops, "mem_gb": mem})
        return len(nodes) - 1

    prev_ffn_ar: Optional[int] = None
    for layer in range(model.num_layers):
        attn_ids = [add(layer, "attn", h, attn_flops / num_kv, attn_mem)
                    for h in range(num_kv)]
        attn_ar = add(layer, "attn_ar", -1, 0.0, 0.0)
        ffn_ids = [add(layer, "ffn", s, ffn_flops / num_ffn, ffn_mem)
                   for s in range(num_ffn)]
        ffn_ar = add(layer, "ffn_ar", -1, 0.0, 0.0)

        for a in attn_ids:
            if prev_ffn_ar is not None:           # layer input feeds each head
                edges.append((prev_ffn_ar, a, act_bytes))
            edges.append((a, attn_ar, act_bytes))  # partial -> attn all-reduce
        for f in ffn_ids:
            edges.append((attn_ar, f, act_bytes))  # attended -> each FFN shard
            edges.append((f, ffn_ar, act_bytes))   # partial -> ffn all-reduce
        prev_ffn_ar = ffn_ar

    return nodes, edges


# ---------------------------------------------------------------------------
# SCT favorite-child LP (Hanen-Munier)
# ---------------------------------------------------------------------------
def _favorite_children(
    nodes: List[Dict], edges: List[Tuple[int, int, float]],
    ref_flops: float, ref_comm_per_byte: float, ref_rtt_s: float,
) -> Dict[int, int]:
    """Solve the SCT LP and round to a favorite-child matching.

    LP (minimise makespan):
        min  Cmax
        s.t. Cmax >= C_i,            C_i >= t_i
             C_j  >= C_i + t_j + c_ij*(1 - x_ij)     for each edge (i,j)
             sum_j x_ij <= 1  (<=1 favorite successor),  sum_i x_ij <= 1
             0 <= x_ij <= 1
    Returns ``{u: v}`` for each favorite edge u->v.
    """
    t = [n["flops"] / ref_flops for n in nodes]                      # ref compute time
    c = [b * ref_comm_per_byte + ref_rtt_s for (_, _, b) in edges]   # ref comm time

    prob = pulp.LpProblem("sct", pulp.LpMinimize)
    C = [pulp.LpVariable(f"C{i}", lowBound=0) for i in range(len(nodes))]
    Cmax = pulp.LpVariable("Cmax", lowBound=0)
    x = [pulp.LpVariable(f"x{e}", lowBound=0, upBound=1) for e in range(len(edges))]

    prob += Cmax
    for i in range(len(nodes)):
        prob += Cmax >= C[i]
        prob += C[i] >= t[i]
    for e, (u, v, _) in enumerate(edges):
        prob += C[v] >= C[u] + t[v] + c[e] * (1 - x[e])
    out_edges: Dict[int, List[int]] = {}
    in_edges: Dict[int, List[int]] = {}
    for e, (u, v, _) in enumerate(edges):
        out_edges.setdefault(u, []).append(e)
        in_edges.setdefault(v, []).append(e)
    for u, es in out_edges.items():
        prob += pulp.lpSum(x[e] for e in es) <= 1
    for v, es in in_edges.items():
        prob += pulp.lpSum(x[e] for e in es) <= 1

    prob.solve(_SOLVER)

    # Round: take edges by descending x, keep a matching (<=1 fav succ/pred).
    order = sorted(range(len(edges)), key=lambda e: -(pulp.value(x[e]) or 0.0))
    fav: Dict[int, int] = {}
    used_pred: set = set()
    for e in order:
        u, v, _ = edges[e]
        if (pulp.value(x[e]) or 0.0) <= 1e-6:
            break
        if u not in fav and v not in used_pred:
            fav[u] = v
            used_pred.add(v)
    return fav


# ---------------------------------------------------------------------------
# Memory-constrained list scheduling (m-ETF with favorite-child preference)
# ---------------------------------------------------------------------------
def _schedule(
    nodes: List[Dict], edges: List[Tuple[int, int, float]],
    fav: Dict[int, int], devices: List[Device], net: Network,
    seq_len: int = 1,
) -> Tuple[List[int], float, bool]:
    """Place each op (nodes are already topologically ordered) and return
    ``(node_device_idx, makespan_s, memory_exceeded)``.

    For each op: co-locate with its favorite predecessor if that device has room
    (zero comm on that arc), else pick the earliest-finish device with free DRAM
    (ETF); if none fit, fall back to the device with the most free memory.

    Concurrency model (so cross-device "parallelism" is not free):
      * **cross-device is unbounded** — every device is an independent worker, so
        ops on different devices overlap freely.
      * **per-device throughput** — a device processes its assigned ops at its
        ``gflops`` (which is the *calibrated multi-core* rate, i.e. ``cores`` are
        already baked in). Running a device's ops concurrently across its cores
        conserves throughput, so it does not change the makespan — the device's
        own ops effectively serialise at ``gflops``.
      * the **coordinator is a serial resource** — in CLIENT_SERVER every
        inter-device transfer is relayed through the one driver/AP, so transfers
        cannot overlap (a single ``coord_free_at`` clock). This is the only
        cluster-level serialiser, so a layer spread across devices pays its
        scatter/gather **serially through the coordinator** — which is why
        spreading does not help over a slow AP (matches the RPi measurements).
        Co-located (favorite-child) edges need no transfer and so cost nothing.
    """
    preds: Dict[int, List[Tuple[int, float]]] = {}
    for (u, v, b) in edges:
        preds.setdefault(v, []).append((u, b))
    fav_pred = {v: u for u, v in fav.items()}

    serial_coord = net.communication_model == CommunicationModel.CLIENT_SERVER

    node_dev = [-1] * len(nodes)
    node_finish = [0.0] * len(nodes)
    dev_mem_used = [0.0] * len(devices)
    dev_free_at = [0.0] * len(devices)   # device throughput is serial (gflops)
    coord_free_at = [0.0]                # serial coordinator clock (boxed)
    mem_exceeded = False

    def evaluate(n: int, d: int) -> Tuple[float, float]:
        """Return (finish_time, coord_free_after) for placing op n on device d,
        given current state — transfers serialise on the coordinator clock; the
        device computes at its (multi-core) gflops throughput."""
        ready = 0.0
        coord = coord_free_at[0]
        pred_list = preds.get(n, [])
        if nodes[n]["split"] == -1 and pred_list:
            # All-reduce node: it reduces the partial tensors of ALL its shards.
            # Price it as a REAL centralized all-reduce over the distinct
            # participant devices (the RPC set/add/get sequence through a leader),
            # NOT a sum of cheap point-to-point sends — otherwise spreading a
            # layer looks far cheaper than the hardware all-reduce. Fewer
            # participants (favorite-child co-location) => cheaper, preserving
            # SCT's benefit; a fully co-located AR (one device) costs nothing.
            ready = max(node_finish[p] for (p, _) in pred_list)
            part = {node_dev[p] for (p, _) in pred_list if node_dev[p] >= 0}
            part.add(d)
            if len(part) > 1:
                ar = net.allreduce_communication_cost(
                    pred_list[0][1], [devices[i] for i in part]
                )
                if serial_coord:                      # occupies the serial coordinator
                    coord = max(ready, coord) + ar
                    ready = coord
                else:                                 # peer-to-peer: no shared queue
                    ready += ar
        else:
            for (p, b) in pred_list:
                if node_dev[p] == d:                  # co-located: no transfer
                    ready = max(ready, node_finish[p])
                    continue
                xfer = net.xfer_time_s(b, devices[node_dev[p]].name, devices[d].name)
                if serial_coord:                      # queue behind other transfers
                    t0 = max(node_finish[p], coord)
                    coord = t0 + xfer
                    ready = max(ready, coord)
                else:                                 # direct peer link, no shared queue
                    ready = max(ready, node_finish[p] + xfer)
        start = max(ready, dev_free_at[d])            # data ready + device free
        # Roofline per-op time (compute vs the op's weight-streaming bandwidth),
        # not flops/gflops — so bandwidth-bound decode is modelled. Op mem is its
        # own weights; the device's ops serialise so per-op terms sum correctly.
        return start + devices[d].compute_time_s(
            nodes[n]["flops"], nodes[n]["mem_gb"], seq_len=seq_len
        ), coord

    for n in range(len(nodes)):
        mem = nodes[n]["mem_gb"]
        fits = [d for d in range(len(devices))
                if dev_mem_used[d] + mem <= devices[d].memory_gb]

        fp = fav_pred.get(n)
        if fp is not None and node_dev[fp] in fits:     # co-locate with favorite
            chosen = node_dev[fp]
        elif fits:                                      # else ETF among feasible
            chosen = min(fits, key=lambda d: evaluate(n, d)[0])
        else:                                           # nothing fits: best effort
            mem_exceeded = True
            chosen = max(range(len(devices)),
                         key=lambda d: devices[d].memory_gb - dev_mem_used[d])

        finish, coord_after = evaluate(n, chosen)
        node_dev[n] = chosen
        node_finish[n] = finish
        dev_mem_used[chosen] += mem
        dev_free_at[chosen] = finish
        coord_free_at[0] = coord_after

    return node_dev, max(node_finish), mem_exceeded


# ---------------------------------------------------------------------------
def simulate_msct(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """Baechi m-SCT placement of the transformer operator DAG.

    Per-token latency is the schedule makespan (critical path through the DAG
    under device contention and inter-device communication) plus the coordinator
    round-trip; total scales by ``gen_tokens``. ``planning_time_s`` is the cost
    of the LP + scheduling.
    """
    t0 = time.perf_counter()
    if not net.servers:
        raise ValueError("No devices available in the network.")
    devices = net.servers
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len

    nodes, edges = _build_op_dag(model, net, seq_len, mem_seq_len)

    # Reference costs for the LP (a representative device / link).
    ref_flops = (sum(d.gflops for d in devices) / len(devices)) * 1e9
    if len(devices) > 1:
        ref_bw, ref_rtt = net.link(devices[0].name, devices[1].name)
    else:
        ref_bw, ref_rtt = float("inf"), 0.0
    ref_comm_per_byte = 0.0 if ref_bw == float("inf") else 8.0 / (ref_bw * 1e6)
    ref_rtt_s = ref_rtt / 2.0 / 1000.0

    fav = _favorite_children(nodes, edges, ref_flops, ref_comm_per_byte, ref_rtt_s)
    node_dev, makespan, mem_exceeded = _schedule(
        nodes, edges, fav, devices, net, seq_len=seq_len
    )

    # Coordinator injection/extraction (entry = device of layer-0's first head,
    # exit = device of the last node).
    coord_io = net.coordinator_io_time_s(
        model.activation_bytes * seq_len,
        model.activation_bytes,  # per-token result egress (small), not full logits
        devices[node_dev[0]].name,
        devices[node_dev[-1]].name,
    )
    # lm_head / output embedding streamed once per token on the exit device, so
    # all policies stream the full model (single_node/pipeline already do).
    lm_head_s = devices[node_dev[-1]].compute_time_s(0.0, model.embedding_weights_gb())
    per_token = makespan + coord_io + lm_head_s
    total = per_token * gen_tokens
    planning_time_s = time.perf_counter() - t0

    # Per-op placement (compute ops only) for the workload-share view + viz.
    placement = [
        {"layer": nodes[i]["layer"], "block_type": nodes[i]["kind"],
         "split": nodes[i]["split"], "device": devices[node_dev[i]].name,
         "flops": nodes[i]["flops"]}
        for i in range(len(nodes)) if nodes[i]["kind"] in ("attn", "ffn")
    ]
    device_memory_gb: Dict[str, float] = {d.name: 0.0 for d in devices}
    for i, nd in enumerate(nodes):
        device_memory_gb[devices[node_dev[i]].name] += nd["mem_gb"]
    n_devices_used = len({p["device"] for p in placement})
    LOGGER.info(
        "m-SCT: %d ops across %d/%d devices, %d favorite-child co-locations, "
        "makespan %.4fs%s",
        len(placement), n_devices_used, len(devices), len(fav), makespan,
        " (memory exceeded)" if mem_exceeded else "",
    )

    _draw_msct_partitions(model, net, node_dev, nodes, seq_len)

    return {
        "policy": "msct",
        "devices": [d.name for d in devices],
        "devices_used": n_devices_used,
        "favorite_colocations": len(fav),
        "memory_exceeded": mem_exceeded,
        "placement": placement,
        "device_memory_gb": device_memory_gb,
        "per_token_s": per_token,
        "total_latency_s": total,
        "planning_time_s": planning_time_s,
        "notes": (
            "Baechi m-SCT: favorite-child LP (Hanen-Munier) over the operator "
            "DAG + memory-constrained ETF scheduling."
        ),
    }


def _draw_msct_partitions(
    model: ModelSpec, net: Network, node_dev: List[int], nodes: List[Dict],
    seq_len: int,
) -> None:
    """Partition graph + device breakdown for the m-SCT placement (gated by
    graph_plots SAVE/SHOW flags). Maps each compute op onto the corresponding
    build_computation_graph node id and colours it by its placed device."""
    attn_split, ffn_split = AttentionSplitStrategy.KV_HEADS, FFNSplitStrategy.KV_HEADS
    xadj, adjncy, edge_weights, _, _, node_weights = model.build_computation_graph(
        net, seq_len, attn_split, ffn_split
    )
    max_attn = model.get_max_attn_splits(attn_split)
    max_ffn = model.get_max_ffn_splits(ffn_split, len(net.servers))
    per_layer_nodes = max_attn + max_ffn + 2

    partitions: List[List[Dict]] = [[] for _ in range(len(net.servers))]
    for i, nd in enumerate(nodes):
        if nd["kind"] not in ("attn", "ffn"):
            continue
        base = nd["layer"] * per_layer_nodes
        nid = base + nd["split"] if nd["kind"] == "attn" else base + max_attn + nd["split"]
        partitions[node_dev[i]].append({
            "node_id": nid, "node_name": f"layer_{nd['layer']}_{nd['kind']}_{nd['split']}",
            "layer": nd["layer"], "block_type": nd["kind"], "split_id": nd["split"],
            "compute_weight": node_weights[nid],
        })

    output_dir = setup_output_dir("sim_output")
    draw_partitions(
        model, net, partitions=partitions, node_weights=node_weights,
        xadj=xadj, adjncy=adjncy, edge_weights=edge_weights,
        attn_split_strategy=attn_split, ffn_split_strategy=ffn_split,
        figsize=(40, 6), save_path=f"{output_dir}/msct_partition_graph.pdf",
        policy_name="m-SCT",
    )
    draw_device_computation_breakdown(
        partitions, net.servers,
        save_path=f"{output_dir}/msct_device_computation_breakdown.pdf",
    )
