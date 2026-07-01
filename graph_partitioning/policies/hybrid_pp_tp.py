"""Memory-aware hybrid pipeline+tensor parallelism heuristic for single-request inference.

NOTE: this is NOT the SCT / m-SCT algorithm of Hanen-Munier / Baechi (that is in
``msct.py``). It is a greedy, transformer-layer-granularity hybrid. It selects a
device subset greedily so that each newly added device has the highest combined
compute/memory/proximity score relative to already-chosen devices, and stops
once the cumulative memory of chosen devices covers the model. Because
tensor-splitting adds an all-reduce per layer, it splits a layer ONLY where
memory forces it:
stages are formed **memory-aware** (``_form_stages``) — a device that fits a
whole layer becomes a solo pipeline stage (no all-reduce), and only devices too
small to hold a layer are bundled into the smallest tensor groups that can. On
a heterogeneous cluster this yields MIXED stages with a *per-stage* tensor
degree, so big devices pipeline whole layers while only the small ones split.
Layers are pipelined across stages in proportion to stage compute.

End-to-end latency uses the same single-stream serial-traversal model as
``simulate_pipeline``: sum of stage compute (plus any intra-group all-reduce) +
inter-stage boundary transfers + coordinator round-trip, times generation
tokens. When every device fits a whole layer, every stage is solo and this
collapses exactly to the pure-pipeline model.
"""

from typing import Dict, List, Optional

from ..common import (
    AttentionSplitStrategy,
    Device,
    FFNSplitStrategy,
    total_avail_memory,
)
from ..graph_plots import (
    draw_device_computation_breakdown,
    draw_partitions,
)
import os
import time

from ..logging_utils import setup_logger, setup_output_dir
from ..model import ModelSpec
from ..network import Network

os.makedirs("sim_output", exist_ok=True)
LOGGER = setup_logger(name=__name__, log_file="sim_output/hybrid_pp_tp.log")


def _proportional_layer_assignment(
    weights: List[float], num_layers: int
) -> List[List[int]]:
    """Allocate ``num_layers`` contiguously across devices in proportion to ``weights``.

    Always assigns at least one layer per device (when num_layers >= num_devs).
    Layers are kept contiguous so it inherits pipeline's communication pattern.
    """
    k = len(weights)
    if k == 0 or num_layers == 0:
        return [[] for _ in range(k)]

    total_w = sum(weights)
    if total_w <= 0:
        # Equal split fallback
        base = num_layers // k
        rem = num_layers % k
        per_dev = [base + (1 if i < rem else 0) for i in range(k)]
    else:
        raw = [w / total_w * num_layers for w in weights]
        per_dev = [max(1, int(r)) if num_layers >= k else int(round(r)) for r in raw]
        # Fix rounding so counts sum to num_layers
        diff = num_layers - sum(per_dev)
        if diff > 0:
            order = sorted(range(k), key=lambda i: raw[i] - per_dev[i], reverse=True)
            for i in order[:diff]:
                per_dev[i] += 1
        elif diff < 0:
            order = sorted(
                range(k), key=lambda i: per_dev[i] - raw[i], reverse=True
            )
            for i in order[: -diff]:
                if per_dev[i] > 1:
                    per_dev[i] -= 1

    parts: List[List[int]] = []
    cursor = 0
    for count in per_dev:
        parts.append(list(range(cursor, cursor + count)))
        cursor += count
    return parts


def _form_stages(
    devices: List[Device], per_layer_mem_gb: float, max_split: int
) -> List[List[Device]]:
    """Group selected devices into pipeline stages, **memory-aware**.

    This heuristic minimises communication, so it tensor-splits a layer ONLY where the
    device it lands on can't hold it. A device that fits a whole layer
    (``per_layer_mem_gb <= memory_gb``) becomes its own pipeline stage — no
    all-reduce. Devices too small to hold a layer are bundled, in selection
    order, into the smallest tensor groups that can: a layer split ``g`` ways
    must fit every member, capped at ``max_split`` (attention can't split finer
    than its KV heads).

    The result is a sequence of stages with a *per-stage* tensor degree: 1 for a
    solo device, ``len(group)`` for a tensor group. In a heterogeneous cluster
    this yields MIXED stages — big devices pipeline whole layers while only the
    small ones pay an intra-group all-reduce. If every device fits a whole layer
    this returns one solo stage per device (pure pipeline). A trailing small
    group that still doesn't fit even at ``max_split`` is returned as-is and
    absorbs the shortfall as a swap penalty downstream.
    """
    # A device fits a whole layer => its own pipeline stage; the rest must be
    # tensor-grouped. Separating them (rather than scanning in order) avoids
    # orphaning a half-built small group when a big device appears between two
    # small ones.
    big = [d for d in devices if d.memory_gb >= per_layer_mem_gb]
    small = [d for d in devices if d.memory_gb < per_layer_mem_gb]

    stages: List[List[Device]] = [[d] for d in big]

    group: List[Device] = []
    for d in small:
        group.append(d)
        fits = per_layer_mem_gb / len(group) <= min(x.memory_gb for x in group)
        if fits or len(group) >= max_split:
            stages.append(list(group))
            group = []
    if group:                       # trailing smalls that don't fit even maxed;
        stages.append(list(group))  # absorbed as a swap penalty downstream
    return stages


def _even_index_chunks(n_items: int, k_parts: int) -> List[range]:
    """Split ``range(n_items)`` into ``k_parts`` near-even contiguous ranges."""
    base, rem = divmod(n_items, k_parts)
    chunks: List[range] = []
    cur = 0
    for i in range(k_parts):
        size = base + (1 if i < rem else 0)
        chunks.append(range(cur, cur + size))
        cur += size
    return chunks


def simulate_hybrid_pp_tp(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """Greedy memory-aware hybrid pipeline+tensor partitioning for wireless heterogeneous devices.

    ``kv_cache_seq_len`` controls the KV-cache length used in memory-pressure
    decisions (device selection + per-layer swap penalty). Defaults to
    ``seq_len``; set to ``prompt_len + decoded_so_far`` to simulate a
    decode step against a long context.
    """

    _t0 = time.perf_counter()
    LOGGER.info("Starting hybrid_pp_tp partitioning")
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len

    chosen_devs: List[Device] = []
    avail_devs = net.filter_servers_by_memory(
        model.attn_memory_gb_per_kv_head(mem_seq_len), net.servers
    )

    while (
        avail_devs
        and total_avail_memory(chosen_devs) < model.total_memory_usage(seq_len=mem_seq_len)
    ):
        scores = net.compute_device_scores(avail_devs, chosen_devs)
        best_dev = avail_devs[scores.index(max(scores))]
        chosen_devs.append(best_dev)
        avail_devs.remove(best_dev)

    if not chosen_devs:
        raise RuntimeError(
            "hybrid_pp_tp could not select any device that fits per-layer memory"
        )

    # --- Form pipeline stages, memory-aware (per-stage tensor degree) -------
    # A device that fits a whole layer becomes a solo pipeline stage; only the
    # devices too small to hold a layer are tensor-grouped. In a heterogeneous
    # cluster this gives MIXED stages — some layers pipelined whole, only the
    # ones on small devices split. When every device fits a layer, every stage
    # is solo and this collapses to pure pipeline.
    per_layer_mem = model.memory_usage_per_device(mem_seq_len, num_layers=[1])[0]
    groups = _form_stages(chosen_devs, per_layer_mem, model.num_kv_heads)
    max_tp = max((len(g) for g in groups), default=1)

    # Layers are pipelined across stages, proportional to each stage's compute.
    group_weights: List[float] = [sum(d.gflops for d in g) for g in groups]
    parts = _proportional_layer_assignment(group_weights, model.num_layers)
    LOGGER.info(
        "hybrid_pp_tp: %d stage(s), tensor degrees %s, layer counts %s",
        len(groups),
        [len(g) for g in groups],
        [len(p) for p in parts],
    )

    q_per_kv = model.num_heads // model.num_kv_heads

    # --- Per-group (stage) compute + intra-group all-reduce -----------------
    stage_compute: List[float] = []
    stage_pairs: List = []
    group_shard: List[Dict] = []
    for gi, (group, layer_ids) in enumerate(zip(groups, parts)):
        n_layers = len(layer_ids)
        split = len(group)
        # Even KV-head split across the group (FFN follows the KV allocation).
        kv_per = [
            model.num_kv_heads // split + (1 if j < model.num_kv_heads % split else 0)
            for j in range(split)
        ]
        ffn_frac = [kv / model.num_kv_heads for kv in kv_per]
        q_per = [kv * q_per_kv for kv in kv_per]

        # Each device's total work over the group's layers; swap is charged on
        # the aggregate working set (as in simulate_tensor_parallel).
        per_dev_time: List[float] = []
        for j, dev in enumerate(group):
            shard_flops = model.layer_compute_flops(
                seq_len, q_per[j], kv_per[j], ffn_frac[j]
            )
            shard_mem = model.memory_usage_per_device(
                mem_seq_len,
                num_layers=[1],
                device_kv_counts=[kv_per[j]],
                device_ffn_fractions=[ffn_frac[j]],
            )[0]
            per_dev_time.append(
                dev.compute_time_s(shard_flops * n_layers, shard_mem * n_layers, seq_len=seq_len)
            )
        group_compute = max(per_dev_time) if per_dev_time else 0.0

        # Intra-group all-reduce (attention + FFN, per layer); 0 for a lone
        # device — that is what keeps pure pipeline communication-cheap.
        if split > 1:
            ar_per_layer = net.allreduce_communication_cost(
                model.attn_allreduce_bytes * seq_len, group
            ) + net.allreduce_communication_cost(
                model.ffn_allreduce_bytes * seq_len, group
            )
        else:
            ar_per_layer = 0.0

        stage_compute.append(group_compute + n_layers * ar_per_layer)
        if gi == 0:
            stage_pairs.append(None)
        else:
            stage_pairs.append((groups[gi - 1][-1].name, group[0].name))
        group_shard.append({"kv_per": kv_per, "ffn_frac": ffn_frac, "split": split})

    # Inter-group boundary activation transfer (pipeline handoff between stages).
    act_bytes = model.ffn_allreduce_bytes * seq_len
    stage_xfer: List[float] = []
    for pair in stage_pairs:
        if pair is None:
            stage_xfer.append(0.0)
        else:
            a, b = pair
            stage_xfer.append(net.xfer_time_s(act_bytes, a, b))

    # Single-stream serial traversal (same model as simulate_pipeline): per
    # forward = sum of stage compute (+ intra-group all-reduce) + every
    # inter-stage hop + the coordinator round-trip, NOT the bottleneck-stage
    # max() — token t+1 can't start until token t's logits exist.
    compute_serial = sum(stage_compute)
    interstage_xfer = sum(stage_xfer)
    coord_xfer = net.coordinator_io_time_s(
        model.activation_bytes * seq_len,
        model.activation_bytes,  # per-token result egress (small), not full logits
        groups[0][0].name,
        groups[-1][-1].name,
    )
    # lm_head / output embedding streamed once per token on the exit device, so
    # all policies stream the full model (single_node/pipeline already do).
    lm_head_s = groups[-1][-1].compute_time_s(0.0, model.embedding_weights_gb())
    per_forward = compute_serial + interstage_xfer + coord_xfer + lm_head_s
    total = per_forward * gen_tokens
    planning_time_s = time.perf_counter() - _t0

    # Build partition graph for visualisation
    xadj, adjncy, edge_weights, _, _, node_weights = model.build_computation_graph(
        net, seq_len
    )

    block_to_max_splits = {
        0: model.get_max_attn_splits(attn_split_strategy),
        1: model.get_max_ffn_splits(ffn_split_strategy, len(chosen_devs)),
    }
    # +2 for the per-layer attn_AR and ffn_AR virtual nodes in the graph.
    per_layer_nodes = sum(block_to_max_splits.values()) + 2

    # Assign each group's layer nodes across that group's devices. The attn and
    # FFN split nodes of a layer are partitioned into ``split`` near-even chunks,
    # one per group device — so a g=1 group puts every node on its single device
    # (identical to pure pipeline) and a g>1 group tensor-splits the layer.
    max_attn = block_to_max_splits[0]
    max_ffn = block_to_max_splits[1]

    partitions: List[List[Dict]] = [[] for _ in range(len(net.servers))]
    for group, layer_ids, info in zip(groups, parts, group_shard):
        split = info["split"]
        group_global = [net.servers.index(d) for d in group]
        attn_chunks = _even_index_chunks(max_attn, split)
        ffn_chunks = _even_index_chunks(max_ffn, split)
        for layer_id in layer_ids:
            start = layer_id * per_layer_nodes
            for j, gdev in enumerate(group_global):
                for split_id in attn_chunks[j]:
                    node_id = start + split_id
                    partitions[gdev].append(
                        {
                            "node_id": node_id,
                            "node_name": f"layer_{layer_id}_attn_{split_id}",
                            "layer": layer_id,
                            "block_type": "attn",
                            "split_id": split_id,
                            "compute_weight": node_weights[node_id],
                        }
                    )
                for split_id in ffn_chunks[j]:
                    node_id = start + max_attn + split_id
                    partitions[gdev].append(
                        {
                            "node_id": node_id,
                            "node_name": f"layer_{layer_id}_ffn_{split_id}",
                            "layer": layer_id,
                            "block_type": "ffn",
                            "split_id": split_id,
                            "compute_weight": node_weights[node_id],
                        }
                    )

    output_dir = setup_output_dir("sim_output")
    with open(f"{output_dir}/setup_info.yaml", "w", encoding="utf-8") as f:
        f.write("policy: hybrid_pp_tp\n")
        f.write(f"model: {model.name}\n")
        f.write(f"num_layers: {model.num_layers}\n")
        f.write(f"devices: {[d.name for d in net.servers]}\n")
        f.write(f"chosen_devices: {[d.name for d in chosen_devs]}\n")
        f.write(f"communication_model: {net.communication_model.name}\n")
        f.write(f"max_tensor_degree: {max_tp}\n")
        f.write(f"groups: {[[d.name for d in g] for g in groups]}\n")
        f.write(f"layers_per_group: {[len(p) for p in parts]}\n")
        f.write(f"total_latency_s: {total}\n")

    draw_partitions(
        model,
        net,
        partitions=partitions,
        node_weights=node_weights,
        xadj=xadj,
        adjncy=adjncy,
        edge_weights=edge_weights,
        attn_split_strategy=attn_split_strategy,
        ffn_split_strategy=ffn_split_strategy,
        figsize=(40, 6),
        save_path=f"{output_dir}/hybrid_pp_tp_partition_graph.pdf",
        policy_name="Hybrid PP+TP",
    )

    draw_device_computation_breakdown(
        partitions,
        net.servers,
        save_path=f"{output_dir}/hybrid_pp_tp_device_computation_breakdown.pdf",
    )

    return {
        "policy": "hybrid_pp_tp",
        "chosen_devices": [d.name for d in chosen_devs],
        "max_tensor_degree": max_tp,
        "stages": [
            {
                "devices": [d.name for d in groups[i]],
                "tp_split": group_shard[i]["split"],
                "layers": parts[i],
                "stage_time_s": stage_compute[i],
                "inbound_xfer_s": stage_xfer[i],
            }
            for i in range(len(groups))
        ],
        "per_forward_s": per_forward,
        "total_latency_s": total,
        "planning_time_s": planning_time_s,
        "breakdown": {
            "compute_serial_s": compute_serial,
            "interstage_xfer_s": interstage_xfer,
            "coord_io_s": coord_xfer,
        },
        "notes": (
            "Greedy device selection by compute/memory/RTT score; hybrid "
            "pipeline+tensor: layers pipelined across device groups, each layer "
            "tensor-split within its group only when one layer won't fit a "
            "single device (tp>1); single-stream serial latency model."
        ),
    }
