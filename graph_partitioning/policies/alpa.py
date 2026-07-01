"""Alpa intra-operator parallelism policy for single-request inference.

Faithful to the intra-operator pass of Alpa (Zheng et al., OSDI '22): the
sharding of the operator graph is chosen by an Integer Linear Program that
minimises Σ(per-operator compute + intrinsic collective) + Σ(resharding
communication on edges), subject to one sharding strategy per operator. We
solve the actual ILP with CBC (via PuLP) over the transformer's operator graph.

Scope note: Alpa's *inter*-operator pass (pipeline stage/submesh assignment via
DP) is a throughput optimisation that amortises the pipeline bubble across
microbatches. Single-request autoregressive decode has B=1 (one token in
flight), so that pass degenerates; the part of Alpa that transfers to this
setting is the intra-op ILP, which is what ``simulate_alpa`` implements. The
inter-op search is provided in ``alpa_pipeline_search`` purely to *demonstrate*
that degeneracy: sweeping the microbatch count B shows the makespan-optimal
stage count stays at 1 until B grows large (see design_exploration.ipynb §11).
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import pulp

from ..common import (
    AttentionSplitStrategy,
    CommunicationModel,
    Device,
    FFNSplitStrategy,
)
from ..graph_plots import draw_device_computation_breakdown, draw_partitions
from ..logging_utils import setup_output_dir
from ..model import ModelSpec
from ..network import Network

# Sharding specs of a tensor on a device mesh.
_R = "R"  # replicated: every device holds the whole tensor
_S = "S"  # sharded: each device holds a slice


def _pick_solver():
    """Return an available MILP solver. Prefer HiGHS (native arm64/x86 wheels via
    highspy, in-process) and fall back to PuLP's bundled CBC binary."""
    try:
        highs = pulp.HiGHS(msg=False)
        if highs.available():
            return highs
    except Exception:  # pragma: no cover - HiGHS not installed
        pass
    return pulp.PULP_CBC_CMD(msg=0)


_SOLVER = _pick_solver()


def _pair_bw_matrix(net: Network) -> Dict[Tuple[str, str], float]:
    """Build a full pairwise bandwidth map using net.link()."""
    names = [d.name for d in net.servers]
    bw = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            bw[(a, b)], _ = net.link(a, b)
    return bw


def _cluster_devices_by_bandwidth(
    net: Network, bw_threshold_mbps: float
) -> List[List[int]]:
    """
    Simple connected-components clustering by 'strong links':
    Put an undirected edge if bw(a,b) >= threshold, then take CCs.
    Returns clusters as lists of server indices.
    """

    names = [d.name for d in net.servers]
    name_to_idx = {n: i for i, n in enumerate(names)}
    g = nx.Graph()
    g.add_nodes_from(names)
    for (a, b), bw in _pair_bw_matrix(net).items():
        if bw >= bw_threshold_mbps:
            g.add_edge(a, b, weight=bw)
    comps = list(nx.connected_components(g))
    clusters = [sorted([name_to_idx[n] for n in comp]) for comp in comps if comp]
    # Fallback: if everything fell below threshold, keep one device per cluster
    if not clusters:
        clusters = [[i] for i in range(len(names))]
    return clusters


def _alpa_form_clusters(net: Network, bw_threshold_mbps: float) -> List[List[int]]:
    """
    Return list of clusters (each is a list of device indices).
    CLIENT_SERVER => single cluster containing all devices.
    PEER_TO_PEER => bandwidth-threshold clustering.
    """
    if net.communication_model == CommunicationModel.CLIENT_SERVER:
        return [list(range(len(net.servers)))]
    return _cluster_devices_by_bandwidth(net, bw_threshold_mbps)


# ---------------------------------------------------------------------------
# Intra-operator ILP
# ---------------------------------------------------------------------------


def _build_op_graph(
    model: ModelSpec,
    net: Network,
    devices: List[Device],
    seq_len: int,
    mem_seq_len: int,
    n_layers: Optional[int] = None,
) -> Tuple[List[Dict], List[Tuple[int, int, float]], Callable[[float], float]]:
    """Build the per-token operator graph for ``n_layers`` layers on ``devices``
    (defaults to the whole model). Pass ``n_layers`` to cost one pipeline stage.

    Each transformer layer contributes four matmul operators chained through
    the residual stream: ``attn_qkv`` and ``ffn_in`` are column-type (replicate
    input, produce a sharded output when split), ``attn_out`` and ``ffn_out``
    are row-type (consume a sharded input, all-reduce to a replicated output).
    Every operator carries its candidate sharding strategies with pre-computed
    node costs (compute + intrinsic collective); edges carry the activation
    size so the ILP can charge resharding (an all-gather when a sharded output
    feeds an operator that wants a replicated input).

    Strategies:
      - ``replicate``  (R->R): full compute on every device, no collective;
        only offered when the whole model fits one device's DRAM.
      - ``shard`` col  (R->S): 1/k compute, output left sharded, no collective.
      - ``shard`` row  (S->R): 1/k compute + an all-reduce to replicate output.
    """
    k = len(devices)
    slowest = min(devices, key=lambda d: d.gflops)  # replicate makespan = slowest device
    min_mem = min(d.memory_gb for d in devices)
    # Replicate keeps the whole model on every device — only legal if it fits.
    replicate_ok = model.total_memory_usage(seq_len=mem_seq_len) <= min_mem

    hidden_bytes = float(model.activation_bytes) * seq_len  # d_model activation
    inter_bytes = 4.0 * model.d_ff * seq_len                # FFN intermediate
    attn_flops = model.attn_compute_flops(seq_len)
    ffn_flops = model.ffn_compute_flops(seq_len)
    # Per-layer weight memory, split across the four matmul ops in the same
    # proportion as their FLOPs, so the roofline bandwidth term per op sums to the
    # layer's weight bytes (and over layers, ~the model's).
    attn_w = model.attn_weights_gb_per_kv_head() * model.num_kv_heads
    ffn_w = model.ffn_memory_gb_per_fraction(1.0)

    def collective(byte_count: float) -> float:
        return net.allreduce_communication_cost(byte_count, devices) if k > 1 else 0.0

    # (name, role, flops, produced-activation bytes, weight GB) repeated per layer.
    template = [
        ("attn_qkv", "col", 0.5 * attn_flops, hidden_bytes, 0.5 * attn_w),
        ("attn_out", "row", 0.5 * attn_flops, hidden_bytes, 0.5 * attn_w),
        ("ffn_in", "col", (2.0 / 3.0) * ffn_flops, inter_bytes, (2.0 / 3.0) * ffn_w),
        ("ffn_out", "row", (1.0 / 3.0) * ffn_flops, hidden_bytes, (1.0 / 3.0) * ffn_w),
    ]

    ops: List[Dict] = []
    for layer in range(n_layers if n_layers is not None else model.num_layers):
        for name, role, flops, out_bytes, op_w in template:
            # Roofline per-op time on the bottleneck (slowest) device — bandwidth-
            # bound at decode, compute-bound at prefill — instead of flops/gflops.
            full_t = slowest.compute_time_s(flops, op_w, seq_len=seq_len)
            shard_t = slowest.compute_time_s(flops / k, op_w / k, seq_len=seq_len)
            strategies: List[Dict] = []
            if replicate_ok or k == 1:
                strategies.append(
                    {"label": "replicate", "in": _R, "out": _R, "cost": full_t}
                )
            if k > 1:
                if role == "col":
                    strategies.append(
                        {"label": "shard", "in": _R, "out": _S, "cost": shard_t}
                    )
                else:  # row: all-reduce to reproduce a replicated output
                    strategies.append(
                        {
                            "label": "shard",
                            "in": _S,
                            "out": _R,
                            "cost": shard_t + collective(out_bytes),
                        }
                    )
            if not strategies:  # model doesn't fit a single device and k == 1
                strategies.append(
                    {"label": "replicate", "in": _R, "out": _R, "cost": full_t}
                )
            ops.append(
                {"name": f"L{layer}.{name}", "out_bytes": out_bytes, "strategies": strategies}
            )

    # Chain consecutive operators; the edge carries the producer's activation.
    edges = [(i, i + 1, ops[i]["out_bytes"]) for i in range(len(ops) - 1)]
    return ops, edges, collective


def _solve_sharding_ilp(
    ops: List[Dict],
    edges: List[Tuple[int, int, float]],
    reshard_cost: Callable[[str, str, float], float],
) -> Tuple[float, List[int]]:
    """Solve Alpa's intra-op ILP exactly with CBC.

    Variables: ``s[v][i]`` (operator v uses strategy i, one-hot). For each edge
    a quadratic term ``s[u][i]·s[v][j]·reshard`` is linearised with an auxiliary
    binary ``e`` (e ≤ s_u, e ≤ s_v, e ≥ s_u + s_v − 1). Objective minimises
    Σ node costs + Σ resharding. Returns (objective_s, chosen strategy index per
    operator).
    """
    prob = pulp.LpProblem("alpa_intra_op", pulp.LpMinimize)

    s: List[List[pulp.LpVariable]] = []
    for v, op in enumerate(ops):
        row = [
            pulp.LpVariable(f"s_{v}_{i}", cat="Binary")
            for i in range(len(op["strategies"]))
        ]
        prob += pulp.lpSum(row) == 1  # exactly one strategy per operator
        s.append(row)

    obj = [
        op["strategies"][i]["cost"] * s[v][i]
        for v, op in enumerate(ops)
        for i in range(len(op["strategies"]))
    ]

    # Resharding edge terms (only materialise vars where the cost is non-zero).
    for u, v, byte_count in edges:
        for i, su in enumerate(ops[u]["strategies"]):
            for j, sv in enumerate(ops[v]["strategies"]):
                rc = reshard_cost(su["out"], sv["in"], byte_count)
                if rc <= 0:
                    continue
                e = pulp.LpVariable(f"e_{u}_{v}_{i}_{j}", cat="Binary")
                prob += e <= s[u][i]
                prob += e <= s[v][j]
                prob += e >= s[u][i] + s[v][j] - 1
                obj.append(rc * e)

    prob += pulp.lpSum(obj)
    prob.solve(_SOLVER)

    chosen = [
        max(range(len(s[v])), key=lambda i: pulp.value(s[v][i]) or 0.0)
        for v in range(len(ops))
    ]
    return float(pulp.value(prob.objective) or 0.0), chosen


def _draw_alpa_partitions(
    model: ModelSpec, net: Network, device_names: List[str], seq_len: int
) -> None:
    """Draw the partition graph + device breakdown for the chosen Alpa mesh.

    The intra-op plan shards/replicates uniformly across the mesh, so each chosen
    device is shown with an equal share of every layer's split nodes (mirrors how
    the workload-share view treats Alpa). Saving/showing is gated by graph_plots.
    """
    attn_split, ffn_split = AttentionSplitStrategy.KV_HEADS, FFNSplitStrategy.KV_HEADS
    xadj, adjncy, edge_weights, _, _, node_weights = model.build_computation_graph(
        net, seq_len, attn_split, ffn_split
    )
    chosen = [i for i, d in enumerate(net.servers) if d.name in set(device_names)]
    if not chosen:
        return
    max_attn = model.get_max_attn_splits(attn_split)
    max_ffn = model.get_max_ffn_splits(ffn_split, len(net.servers))
    per_layer_nodes = max_attn + max_ffn + 2  # +2 for the attn_AR / ffn_AR nodes

    def chunks(n_items: int, k_parts: int) -> List[range]:
        base, rem = divmod(n_items, k_parts)
        out, cur = [], 0
        for i in range(k_parts):
            size = base + (1 if i < rem else 0)
            out.append(range(cur, cur + size))
            cur += size
        return out

    attn_chunks = chunks(max_attn, len(chosen))
    ffn_chunks = chunks(max_ffn, len(chosen))
    partitions: List[List[Dict]] = [[] for _ in range(len(net.servers))]
    for layer in range(model.num_layers):
        start = layer * per_layer_nodes
        for j, gdev in enumerate(chosen):
            for sid in attn_chunks[j]:
                nid = start + sid
                partitions[gdev].append({
                    "node_id": nid, "node_name": f"layer_{layer}_attn_{sid}",
                    "layer": layer, "block_type": "attn", "split_id": sid,
                    "compute_weight": node_weights[nid],
                })
            for sid in ffn_chunks[j]:
                nid = start + max_attn + sid
                partitions[gdev].append({
                    "node_id": nid, "node_name": f"layer_{layer}_ffn_{sid}",
                    "layer": layer, "block_type": "ffn", "split_id": sid,
                    "compute_weight": node_weights[nid],
                })

    output_dir = setup_output_dir("sim_output")
    draw_partitions(
        model, net, partitions=partitions, node_weights=node_weights,
        xadj=xadj, adjncy=adjncy, edge_weights=edge_weights,
        attn_split_strategy=attn_split, ffn_split_strategy=ffn_split,
        figsize=(40, 6), save_path=f"{output_dir}/alpa_partition_graph.pdf",
        policy_name="Alpa",
    )
    draw_device_computation_breakdown(
        partitions, net.servers,
        save_path=f"{output_dir}/alpa_device_computation_breakdown.pdf",
    )


def simulate_alpa(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    bw_threshold_mbps: float = 10.0,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """Alpa intra-operator parallelism for single-request inference.

    For each device mesh (cluster) the per-operator sharding is chosen by the
    ILP above; total per-token latency is the ILP objective (compute + intra-op
    collectives + resharding) plus the coordinator round-trip. The best mesh is
    returned. ``planning_time_s`` reports the wall-clock cost of forming clusters
    and solving the ILP(s) — the price of computing this allocation.
    """
    assert net.servers, "No devices available"
    t0 = time.perf_counter()
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len

    # Proximity-aware device ranking: in CLIENT_SERVER every token pays a
    # coordinator round-trip to/from the mesh, so rank devices by their solo cost
    # (compute + coordinator I/O), not raw GFLOPS. The m=1 mesh then picks the
    # device that is fast AND close to the driver (matching single_node); larger
    # meshes take the m best by the same measure. (coord I/O is 0 in PEER_TO_PEER,
    # so this reduces to fastest-compute.)
    _rank_flops = model.layer_compute_flops(seq_len) * model.num_layers
    _rank_mem = model.total_memory_usage(seq_len=mem_seq_len)
    _rank_in = model.activation_bytes * seq_len
    _rank_out = model.activation_bytes

    def _solo_cost(d) -> float:
        # Same measure as single_node's _single_node_cost: roofline + seq-aware
        # R_compute (so the ranking matches how the device will actually run) plus
        # the coordinator round-trip — not a bare flops/gflops.
        return d.compute_time_s(
            _rank_flops, _rank_mem, seq_len=seq_len
        ) + net.coordinator_io_time_s(_rank_in, _rank_out, d.name, d.name)

    clusters = _alpa_form_clusters(net, bw_threshold_mbps)

    best: Optional[float] = None
    best_detail: Optional[Dict] = None

    for cl_idx, cl in enumerate(clusters):
        cl_devs = [net.servers[i] for i in cl]
        # Alpa also chooses the MESH SIZE, not just the per-op sharding: try the
        # top-m fastest devices of this cluster (m = 1..len). On a slow fabric
        # the search keeps m small (sharding's collectives aren't worth it — at
        # m=1 it is a single device, no comm); on a fast one it widens to shard.
        ranked = sorted(cl_devs, key=_solo_cost)  # proximity-aware: fast AND near driver
        for m in range(1, len(ranked) + 1):
            devs = ranked[:m]
            ops, edges, collective = _build_op_graph(
                model, net, devs, seq_len, mem_seq_len
            )

            def _reshard(out_spec: str, in_spec: str, byte_count: float,
                         _c=collective) -> float:
                # Making a sharded tensor replicated costs an all-gather; a
                # replicated tensor feeding a sharded op is a local slice (~free).
                if out_spec == _S and in_spec == _R:
                    return _c(byte_count)
                return 0.0

            intra_op_s, chosen = _solve_sharding_ilp(ops, edges, _reshard)

            # Coordinator injection/extraction (CLIENT_SERVER only); one round-trip.
            coord_io = net.coordinator_io_time_s(
                model.activation_bytes * seq_len,
                model.activation_bytes,  # per-token result egress (small), not full logits
                devs[0].name,
                devs[-1].name,
            )
            # lm_head / output embedding streamed once per token on the exit
            # device, so all policies stream the full model (the op graph above
            # only covers the per-layer matmuls).
            lm_head_s = devs[-1].compute_time_s(0.0, model.embedding_weights_gb())
            per_token = intra_op_s + coord_io + lm_head_s
            total = per_token * gen_tokens

            strategy_counts: Dict[str, int] = {}
            for v, idx in enumerate(chosen):
                label = ops[v]["strategies"][idx]["label"]
                strategy_counts[label] = strategy_counts.get(label, 0) + 1

            if best is None or total < best:
                best = total
                best_detail = {
                    "cluster_index": cl_idx,
                    "devices": [d.name for d in devs],
                    "mesh_size": m,
                    "intra_op_cost_s": intra_op_s,
                    "coord_io_s": coord_io,
                    "per_token_s": per_token,
                    "strategy_counts": strategy_counts,
                    "op_strategies": [ops[v]["strategies"][chosen[v]]["label"]
                                      for v in range(len(ops))],
                }

    assert best_detail is not None

    # Degenerate plan: if the optimum shards NOTHING (every operator replicated),
    # the extra devices are pure redundancy — each recomputes the same activations
    # and none reduces latency (alpa correctly found tensor-sharding's collectives
    # aren't worth it on this fabric). A replicated "mesh" is just one device run
    # N times, so report it as what actually runs: a single device — the fastest-
    # and-nearest one (min solo cost, identical to single_node), not an idle mesh.
    if best_detail["mesh_size"] > 1 and set(best_detail["strategy_counts"]) <= {"replicate"}:
        solo = min(net.servers, key=_solo_cost)
        compute_s = solo.compute_time_s(_rank_flops, _rank_mem, seq_len=seq_len)
        coord_io = net.coordinator_io_time_s(_rank_in, _rank_out, solo.name, solo.name)
        per_token = compute_s + coord_io
        best = per_token * gen_tokens
        best_detail = {
            "cluster_index": best_detail["cluster_index"],
            "devices": [solo.name],
            "mesh_size": 1,
            "intra_op_cost_s": compute_s,
            "coord_io_s": coord_io,
            "per_token_s": per_token,
            "strategy_counts": {"replicate": best_detail["strategy_counts"].get("replicate", 0)},
            "op_strategies": best_detail["op_strategies"],
            "collapsed_to_single_device": True,
        }

    planning_time_s = time.perf_counter() - t0

    # Partition graph + device breakdown for the chosen mesh, like the other
    # policies (honours SAVE_FIGURES / SHOW_FIGURES via graph_plots).
    _draw_alpa_partitions(model, net, best_detail["devices"], seq_len)

    return {
        "policy": "alpa",
        "topology": net.communication_model.value,
        "clusters": [[net.servers[i].name for i in cl] for cl in clusters],
        "selected_plan": best_detail,
        "total_latency_s": best,
        "planning_time_s": planning_time_s,
        "notes": (
            "Alpa intra-operator ILP (CBC): per-operator sharding minimising "
            "compute + intra-op collectives + resharding. Inter-op/pipeline DP "
            "omitted (degenerate at B=1 for single-request decode)."
        ),
    }


# ---------------------------------------------------------------------------
# Inter-operator (pipeline) pass — included to DEMONSTRATE it does not help
# single-request decode. Alpa's inter-op DP slices the graph into stages on
# submeshes and amortises the pipeline bubble across MICROBATCHES (B). With one
# request in flight (B=1) there is nothing to amortise: the GPipe makespan
#     makespan(B) = Σ_s stage_time_s + (B-1) * max_s stage_time_s
# collapses to a pure serial traversal that only grows with more stages (each
# adds a hop). So the throughput-optimal plan at B=1 is a single stage — no
# pipelining. The design_exploration experiment sweeps B to show the optimal
# stage count stay at 1 until B grows large.
# ---------------------------------------------------------------------------


def _intra_op_cost_s(
    model: ModelSpec,
    net: Network,
    devices: List[Device],
    seq_len: int,
    mem_seq_len: int,
    n_layers: Optional[int] = None,
) -> float:
    """Intra-op ILP objective (per-token compute + collectives + resharding) for
    ``n_layers`` layers on ``devices``."""
    ops, edges, collective = _build_op_graph(
        model, net, devices, seq_len, mem_seq_len, n_layers
    )

    def _reshard(out_spec: str, in_spec: str, byte_count: float) -> float:
        return collective(byte_count) if (out_spec == _S and in_spec == _R) else 0.0

    obj, _ = _solve_sharding_ilp(ops, edges, _reshard)
    return obj


def _even_contiguous(items: List, n_parts: int) -> List[List]:
    """Split ``items`` into ``n_parts`` contiguous, near-even chunks."""
    base, rem = divmod(len(items), n_parts)
    out, cur = [], 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        out.append(items[cur : cur + size])
        cur += size
    return out


def alpa_pipeline_search(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    microbatches: int,
    kv_cache_seq_len: Optional[int] = None,
    devices: Optional[List[Device]] = None,
) -> Dict:
    """Alpa's inter-op pass (simplified, single cluster): for each stage count S,
    split the cluster into S contiguous submeshes and the layers into S
    contiguous groups, cost each stage's sharding with the intra-op ILP, and
    evaluate the GPipe makespan for ``microbatches`` microbatches.

    Returns per-S results plus the makespan-optimal stage count. At
    ``microbatches == 1`` the optimum is S=1 — the negative result for
    single-request decode (pipelining only adds serial hops, no bubble to fill).
    """
    devices = list(devices if devices is not None else net.servers)
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    max_stages = min(len(devices), model.num_layers)

    # Per-layer intra-op cost depends on the submesh (assumed homogeneous here),
    # so cache by submesh size and scale by the stage's layer count.
    per_layer_cache: Dict[int, float] = {}

    def per_layer(submesh: List[Device]) -> float:
        key = len(submesh)
        if key not in per_layer_cache:
            per_layer_cache[key] = _intra_op_cost_s(
                model, net, submesh, seq_len, mem_seq_len, n_layers=1
            )
        return per_layer_cache[key]

    results = []
    for n_stages in range(1, max_stages + 1):
        submeshes = _even_contiguous(devices, n_stages)
        layer_groups = _even_contiguous(list(range(model.num_layers)), n_stages)
        stage_compute = [
            per_layer(m) * len(lg) for m, lg in zip(submeshes, layer_groups)
        ]
        # Activation handoff into each stage (relayed via the coordinator in
        # CLIENT_SERVER); stage 0 has no inbound pipeline hop.
        hop_in = [0.0] + [
            net.xfer_time_s(
                model.activation_bytes * seq_len,
                submeshes[s - 1][-1].name,
                submeshes[s][0].name,
            )
            for s in range(1, n_stages)
        ]
        coord_io = net.coordinator_io_time_s(
            model.activation_bytes * seq_len,
            model.activation_bytes,  # per-token result egress (small), not full logits
            submeshes[0][0].name,
            submeshes[-1][-1].name,
        )
        stage_time = [stage_compute[s] + hop_in[s] for s in range(n_stages)]

        per_request_latency = sum(stage_time) + coord_io   # B=1 traversal
        bottleneck = max(stage_time)
        makespan = per_request_latency + (microbatches - 1) * bottleneck
        # GPipe bubble fraction: idle pipeline-slot time / total.
        bubble = (n_stages - 1) / (microbatches + n_stages - 1)

        results.append({
            "stages": n_stages,
            "makespan_s": makespan,
            "per_request_latency_s": per_request_latency,
            "bottleneck_s": bottleneck,
            "throughput_req_s": microbatches / makespan if makespan > 0 else 0.0,
            "bubble_fraction": bubble,
        })

    best = min(results, key=lambda r: r["makespan_s"])
    return {
        "microbatches": microbatches,
        "optimal_stages": best["stages"],
        "results": results,
    }
