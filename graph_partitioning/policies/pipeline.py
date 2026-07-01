"""Simple pipeline parallelism policy for single-request inference."""

import time
from typing import Dict, List, Optional

from ..common import (
    Device,
    AttentionSplitStrategy,
    FFNSplitStrategy,
)
from ..graph_plots import (
    draw_device_computation_breakdown,
    draw_partitions,
)
from ..logging_utils import setup_output_dir
from ..model import ModelSpec
from ..network import Network


def _pipeline_partition_by_compute(
    devices: List[Device], layer_costs: List[float]
) -> List[List[int]]:
    """
    Greedy contiguous partitioning of layers to roughly balance compute across stages.
    Returns a list of lists of layer indices per stage.
    """
    k = len(devices)
    total = sum(layer_costs)
    target = total / k
    parts: List[List[int]] = [[] for _ in range(k)]
    acc = 0.0
    stage = 0
    for i, cost in enumerate(layer_costs):
        if stage < k - 1 and acc + cost > (stage + 1) * target and parts[stage]:
            stage += 1
        parts[stage].append(i)
        acc += cost
    return parts


def simulate_pipeline(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """
    Estimate pipeline-parallel token latency with simple bubble model.
    Assumes each boundary sends activation for each token once.

    ``kv_cache_seq_len`` sets the per-layer KV-cache size used for the swap
    penalty; defaults to ``seq_len``. Set it to ``prompt_len + decoded_so_far``
    when simulating a decode step against a long context.
    """
    _t0 = time.perf_counter()
    devices = net.servers
    layer_compute_cost = model.layer_compute_flops(seq_len)
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    layer_memory_cost = model.memory_usage_per_device(mem_seq_len, num_layers=[1])[0]
    parts = _pipeline_partition_by_compute(
        devices, [layer_compute_cost] * model.num_layers
    )

    # Stage compute times per token
    stage_compute = []
    stage_pairs = []
    for stage_idx, layer_ids in enumerate(parts):
        stage_flops = len(layer_ids) * layer_compute_cost
        stage_mem = len(layer_ids) * layer_memory_cost
        # The token-embedding / output (lm_head) matrix lives on the last stage
        # (the output projection is streamed every decode step). Count it so the
        # total streamed bytes match single_node's total_memory_usage — otherwise
        # pipeline under-streams by the embedding and n>=2 decode comes out faster
        # than n=1, which is non-physical.
        if stage_idx == len(parts) - 1:
            stage_mem += model.embedding_weights_gb()
        t = devices[stage_idx].compute_time_s(stage_flops, stage_mem, seq_len=seq_len)
        stage_compute.append(t)
        if stage_idx > 0:
            a = devices[stage_idx - 1].name
            b = devices[stage_idx].name
            stage_pairs.append((a, b))
        else:
            stage_pairs.append(None)

    # Activation transfer times between stages.
    # At a pipeline boundary we ship the full activation tensor for the whole
    # (micro-)batch: seq_len tokens × d_model × dtype bytes.
    # This is different from a tensor-parallel all-reduce which only syncs a
    # single token's worth of activations.
    act = model.activation_bytes * seq_len  # one hidden state per token, point-to-point
    stage_xfer = []
    for pair in stage_pairs:
        if pair is None:
            stage_xfer.append(0.0)
        else:
            a, b = pair
            stage_xfer.append(net.xfer_time_s(act, a, b))

    # Single-stream autoregressive inference is a *serial traversal*: token
    # t+1 cannot begin until token t's logits are produced, so the pipeline
    # never fills (exactly one forward in flight). The per-forward latency is
    # therefore the SUM of every stage's compute plus every boundary transfer,
    # NOT the bottleneck-stage max() — which is the steady-state throughput of
    # a *full* pipeline carrying many concurrent requests, a different
    # workload. Using the sum is what makes added devices *increase* latency
    # (each extra stage adds a network hop with no parallelism benefit),
    # matching the measured hardware behaviour for both prefill and decode.
    #
    # The seq_len argument makes this phase-aware automatically: prefill is one
    # batched forward with seq_len = prompt_len (compute- and payload-heavy,
    # network amortised over the batch); decode is seq_len = 1 (compute-tiny,
    # so the per-hop network cost dominates).
    compute_serial = sum(stage_compute)        # total compute across all stages
    interstage_xfer = sum(stage_xfer)          # (K-1) inter-stage activation hops

    # Coordinator injection/extraction (input -> stage 0, result <- last stage).
    # The per-forward egress is the SMALL result the driver needs each step, not
    # the full vocab logits: llama.cpp samples on the last stage (or keeps the
    # output head on the driver), so only ~one hidden state / sampled token comes
    # back per token. Shipping full logits (output_allreduce_bytes, ~128 KB) every
    # decode token over the heterogeneous mesh produced a non-physical, link-
    # dependent wiggle (the last stage — and its link to the coordinator — changes
    # with device count) that contradicted the smooth measured tg curve.
    coord_xfer = net.coordinator_io_time_s(
        model.activation_bytes * seq_len,
        model.activation_bytes,
        devices[0].name,
        devices[-1].name,
    )

    forward_latency = compute_serial + interstage_xfer + coord_xfer
    total = forward_latency * gen_tokens
    planning_time_s = time.perf_counter() - _t0

    # Create plotting variables for pipeline visualization
    xadj, adjncy, edge_weights, _, _, node_weights = model.build_computation_graph(
        net, seq_len
    )

    # Default split strategies for pipeline (not used but required for API)
    attn_split_strategy = AttentionSplitStrategy.KV_HEADS
    ffn_split_strategy = FFNSplitStrategy.KV_HEADS

    block_to_max_splits = {
        0: model.get_max_attn_splits(attn_split_strategy),
        1: model.get_max_ffn_splits(ffn_split_strategy, len(net.servers)),
    }
    # +2 for the per-layer attn_AR and ffn_AR virtual nodes in the graph.
    per_layer_nodes = sum(block_to_max_splits.values()) + 2

    # Convert pipeline stages to METIS-style partition format
    partitions = []
    for stage_idx, layer_ids in enumerate(parts):
        stage_partition = []
        for layer_id in layer_ids:
            start_node_id = layer_id * per_layer_nodes
            for block_type, max_splits in block_to_max_splits.items():
                for split_id in range(max_splits):
                    if block_type == 0:
                        node_id = start_node_id + split_id
                    else:
                        node_id = start_node_id + block_to_max_splits[0] + split_id
                    # Create node entries
                    stage_partition.append(
                        {
                            "node_id": node_id,
                            "node_name": f"layer_{layer_id}_{'attn' if block_type == 0 else 'ffn'}_{split_id}",
                            "layer": layer_id,
                            "block_type": ("attn" if block_type == 0 else "ffn"),
                            "split_id": split_id,
                            "compute_weight": node_weights[node_id],
                        }
                    )
        partitions.append(stage_partition)

    output_dir = setup_output_dir("sim_output")
    with open(f"{output_dir}/setup_info.yaml", "w", encoding="utf-8") as f:
        f.write("policy: pipeline\n")
        f.write(f"model: {model.name}\n")
        f.write(f"num_layers: {model.num_layers}\n")
        f.write(f"devices: {[d.name for d in devices]}\n")
        f.write(f"communication_model: {net.communication_model.name}\n")
        f.write(f"total_latency_s: {total}\n")
        f.write(f"stages: {len(parts)}\n")
        f.write(f"layers_per_stage: {[len(stage) for stage in parts]}\n")

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
        save_path=f"{output_dir}/pipeline_partition_graph.pdf",
        policy_name="Pipeline",
    )

    draw_device_computation_breakdown(
        partitions,
        net.servers,
        save_path=f"{output_dir}/pipeline_device_computation_breakdown.pdf",
    )

    return {
        "policy": "pipeline",
        "stages": [
            {
                "device": d.name,
                "layers": parts[i],
                "compute_s": stage_compute[i],
                "inbound_xfer_s": stage_xfer[i],
                "gflops": len(parts[i]) * layer_compute_cost / 1e9,
                "mem_gb": len(parts[i]) * layer_memory_cost,
            }
            for i, d in enumerate(devices)
        ],
        "per_forward_s": forward_latency,
        "total_latency_s": total,
        "planning_time_s": planning_time_s,
        "breakdown": {
            "compute_serial_s": compute_serial,
            "interstage_xfer_s": interstage_xfer,
            "coord_io_s": coord_xfer,
        },
        "notes": (
            "Single-stream serial traversal: per-forward latency = sum of stage "
            "compute + inter-stage hops + coordinator round-trip."
        ),
    }
