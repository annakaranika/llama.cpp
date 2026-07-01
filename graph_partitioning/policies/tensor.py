"""Simple Tensor parallelism policy for single-request inference."""

import time
from typing import Dict, Optional

from ..graph_plots import (
    draw_device_computation_breakdown,
    draw_partitions,
)
from ..logging_utils import setup_output_dir
from ..model import ModelSpec
from ..network import Network
from ..common import (
    AttentionSplitStrategy,
    FFNSplitStrategy,
    std_times,
    efficiency,
)


def simulate_tensor_parallel(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    parallel_degree: Optional[int] = None,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """
    Tensor-parallel over a group of devices with flexible head-level splitting strategies
    for attention and FFN.
    No KV head replication - limited by num_kv_heads.

    Supported combinations:
    1. attn: kv_heads, ffn: kv_heads     -> Both split by KV head groups
    2. attn: kv_heads, ffn: hidden_dim   -> Attention by KV heads, FFN by dimension
    3. attn: kv_heads, ffn: q_heads      -> Attention by KV heads, FFN by Q head groups.
    """
    # Cap the tensor-parallel degree at ``num_kv_heads`` because attention is
    # split by KV-head groups with no replication. If the cluster has more
    # devices than KV heads, the surplus devices simply don't participate;
    # we pick the fastest k by GFLOPS so the surviving group is as strong as
    # possible (matters when the user supplies a heterogeneous server list).
    _t0 = time.perf_counter()
    requested_k = (
        len(net.servers)
        if parallel_degree is None
        else min(parallel_degree, len(net.servers))
    )
    k = min(requested_k, model.num_kv_heads)
    if k < 1:
        raise ValueError(
            f"Cannot run tensor parallel with num_kv_heads={model.num_kv_heads}"
        )

    if k < requested_k:
        # Pick the k strongest servers; preserve their original order so the
        # network's pairwise-link assumptions stay consistent.
        ranked = sorted(
            enumerate(net.servers), key=lambda iv: iv[1].gflops, reverse=True
        )
        chosen_indices = sorted(i for i, _ in ranked[:k])
        devices = [net.servers[i] for i in chosen_indices]
    else:
        if parallel_degree is not None:
            chosen_indices = list(range(k))
            devices = net.servers[:k]
        else:
            chosen_indices = list(range(len(net.servers)))
            devices = net.servers

    # Attention is always split by KV heads (no replication)
    kv_heads_per_device = model.num_kv_heads // k
    remaining_kv = model.num_kv_heads % k

    device_kv_counts = [kv_heads_per_device] * k
    for i in range(remaining_kv):
        device_kv_counts[i] += 1

    # Q heads follow KV heads (GQA pattern: each KV head serves multiple Q heads)
    q_heads_per_kv = model.num_heads // model.num_kv_heads
    device_q_counts = [kv * q_heads_per_kv for kv in device_kv_counts]

    if attn_split_strategy == AttentionSplitStrategy.KV_HEADS:
        # Attention splits follow KV head allocation
        device_attn_fractions = [
            kv_count / model.num_kv_heads for kv_count in device_kv_counts
        ]
    else:
        raise ValueError(f"Unsupported attention split strategy: {attn_split_strategy}")

    # FFN split strategy determines workload distribution
    if ffn_split_strategy == FFNSplitStrategy.KV_HEADS:
        # FFN splits follow KV head allocation
        device_ffn_fractions = [
            kv_count / model.num_kv_heads for kv_count in device_kv_counts
        ]
    elif ffn_split_strategy == FFNSplitStrategy.HIDDEN_DIM:
        # FFN splits evenly across hidden dimension (traditional tensor parallel)
        device_ffn_fractions = [1.0 / k] * k
    elif ffn_split_strategy == FFNSplitStrategy.Q_HEADS:
        # FFN splits follow Q head allocation
        device_ffn_fractions = [
            q_count / model.num_heads for q_count in device_q_counts
        ]
    else:
        raise ValueError(f"Unsupported FFN split strategy: {ffn_split_strategy}")

    # Calculate per-device compute + swap penalty across all layers.
    # The previous implementation called ``compute_time_s`` with only
    # *per-layer* memory and then multiplied compute by num_layers, which
    # silently ignored swap pressure: each layer's shard fits in DRAM in
    # isolation, but the stacked working set of num_layers shards does not.
    # We aggregate flops AND memory before the swap check so the per-device
    # forward-pass time correctly includes any disk paging.
    max_per_layer_flops = 0.0
    per_layer_compute_times = []
    device_memory_usage = []
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    for i, dev in enumerate(devices):
        per_layer_compute_flops = model.layer_compute_flops(
            seq_len,
            device_q_counts[i],
            device_kv_counts[i],
            device_ffn_fractions[i],
        )

        # Per-layer memory shard (for reporting in the result dict)
        per_layer_mem = model.memory_usage_per_device(
            mem_seq_len,
            num_layers=[1],
            device_kv_counts=device_kv_counts[i : i + 1],
            device_ffn_fractions=device_ffn_fractions[i : i + 1],
        )[0]
        device_memory_usage.append(per_layer_mem)

        # Total per-device working set across all layers' shards.
        total_dev_flops = per_layer_compute_flops * model.num_layers
        total_dev_mem = per_layer_mem * model.num_layers
        # compute_time_s charges swap based on total_dev_mem vs DRAM, so the
        # returned time is "compute for all num_layers passes + any swap I/O".
        per_dev_time = dev.compute_time_s(total_dev_flops, total_dev_mem, seq_len=seq_len)
        # Store the per-layer-equivalent (divide out num_layers) so the rest
        # of the code, which still treats ``per_layer_compute_times`` as a
        # per-layer quantity, can keep its existing ``× num_layers`` math.
        per_layer_compute_times.append(per_dev_time / model.num_layers)
        max_per_layer_flops = max(per_layer_compute_flops, max_per_layer_flops)
    compute_time = max(per_layer_compute_times) * model.num_layers

    # All-reduce cost per layer (assume 2 collectives per layer).
    # During prefill the all-reduce syncs seq_len tokens simultaneously,
    # so the payload is seq_len × the per-token byte count.

    # Attention all-reduce: needed after attention computation
    attn_devices = [
        device
        for device, attn_frac in zip(devices, device_attn_fractions)
        if attn_frac > 0
    ]
    attn_comm_cost = net.allreduce_communication_cost(
        model.attn_allreduce_bytes * seq_len, attn_devices
    )

    # FFN all-reduce: needed after FFN computation
    ffn_devices = [
        device
        for device, ffn_frac in zip(devices, device_ffn_fractions)
        if ffn_frac > 0
    ]
    ffn_comm_cost = net.allreduce_communication_cost(
        model.ffn_allreduce_bytes * seq_len, ffn_devices
    )

    # Total communication per layer: attention + FFN collectives
    per_layer_comm = attn_comm_cost + ffn_comm_cost
    comm_time = per_layer_comm * model.num_layers

    # Coordinator injection/extraction (input -> shard group, logits <- group),
    # CLIENT_SERVER only. One round-trip per forward, on top of the per-layer
    # all-reduces. Uses the participating devices' endpoints.
    coord_io = net.coordinator_io_time_s(
        model.activation_bytes * seq_len,
        model.activation_bytes,  # per-token result egress (small), not full logits
        devices[0].name,
        devices[-1].name,
    )

    # lm_head / output embedding is streamed once per token on the exit device —
    # counted here so all policies stream the full model (single_node/pipeline
    # already include it via total_memory_usage / the last stage).
    lm_head_s = devices[-1].compute_time_s(0.0, model.embedding_weights_gb())
    per_token = compute_time + comm_time + coord_io + lm_head_s
    total = per_token * gen_tokens

    # Calculate load balance and memory metrics
    compute_time_std = std_times(per_layer_compute_times)
    load_balance_efficiency_val = efficiency(per_layer_compute_times)
    planning_time_s = time.perf_counter() - _t0

    # Create plotting variables for tensor parallelism visualization

    xadj, adjncy, edge_weights, _, _, node_weights = model.build_computation_graph(
        net, seq_len, attn_split_strategy, ffn_split_strategy
    )

    block_type_to_fractions = {
        0: device_attn_fractions,
        1: device_ffn_fractions,
    }
    block_type_to_max_splits = {
        0: model.get_max_attn_splits(attn_split_strategy),
        1: model.get_max_ffn_splits(ffn_split_strategy, k),
    }
    # +2 for the per-layer attn_AR and ffn_AR virtual nodes in the graph.
    per_layer_nodes = sum(block_type_to_max_splits.values()) + 2
    # Partitions are indexed by the original net.servers slot so the
    # visualisation colours and legend stay aligned even when k <
    # len(net.servers). Unused server slots stay empty.
    partitions = [[] for _ in range(len(net.servers))]
    for block_type, dev_fracs in block_type_to_fractions.items():
        sum_splits = 0
        for dev_id, dev_frac in enumerate(dev_fracs):
            node_num = int(block_type_to_max_splits[block_type] * dev_frac)
            global_dev_id = chosen_indices[dev_id]
            for split_id in range(node_num):
                for layer_id in range(model.num_layers):
                    node_id = layer_id * per_layer_nodes + sum_splits + split_id
                    if block_type == 1:
                        node_id += block_type_to_max_splits[0]
                    block_name = "attn" if block_type == 0 else "ffn"
                    node_name = f"layer_{layer_id}_{block_name}_{sum_splits + split_id}"
                    partitions[global_dev_id].append(
                        {
                            "node_id": node_id,
                            "node_name": node_name,
                            "layer": layer_id,
                            "block_type": block_name,
                            "split_id": split_id,
                            "compute_weight": node_weights[node_id],
                        }
                    )
            sum_splits += node_num

    # Create output directory and save metadata
    output_dir = setup_output_dir("sim_output")
    with open(f"{output_dir}/setup_info.yaml", "w", encoding="utf-8") as f:
        f.write("policy: tensor\n")
        f.write(f"model: {model.name}\n")
        f.write(f"num_layers: {model.num_layers}\n")
        f.write(f"devices: {[d.name for d in devices]}\n")
        f.write(f"communication_model: {net.communication_model.name}\n")
        f.write(f"total_latency_s: {total}\n")
        f.write(f"parallel_degree: {k}\n")
        f.write(f"attention_split_strategy: {attn_split_strategy.name}\n")
        f.write(f"ffn_split_strategy: {ffn_split_strategy.name}\n")

    # Generate visualization plots
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
        save_path=f"{output_dir}/tensor_parallel_partition_graph.pdf",
        policy_name="Tensor",
    )

    draw_device_computation_breakdown(
        partitions,
        net.servers,
        save_path=f"{output_dir}/tensor_parallel_device_computation_breakdown.pdf",
    )

    return {
        "policy": "tensor",
        "communication_model": net.communication_model.value,
        "k": k,
        "split_strategies": {
            "attn": attn_split_strategy,
            "ffn": ffn_split_strategy,
        },
        "device_allocation": {
            devices[i].name: {
                "q_heads": device_q_counts[i],
                "kv_heads": device_kv_counts[i],
                "ffn_fraction": device_ffn_fractions[i],
                "memory_usage_gb": device_memory_usage[i],
                "compute_time_per_layer": per_layer_compute_times[i],
            }
            for i in range(k)
        },
        "per_token_s": per_token,
        "per_layer_max_gflops": max_per_layer_flops / 1e9,
        "component_times_s": {
            "compute": compute_time,
            "communication": comm_time,
            "attn_communication": attn_comm_cost * model.num_layers,
            "ffn_communication": ffn_comm_cost * model.num_layers,
        },
        "efficiency_metrics": {
            "load_balance_efficiency": load_balance_efficiency_val,
            "compute_time_std": compute_time_std,
        },
        "total_latency_s": total,
        "planning_time_s": planning_time_s,
        "notes": (
            f"Using {net.communication_model.value} communication with pairwise link modeling. "
            f"Attention split by {attn_split_strategy}, FFN split by {ffn_split_strategy}. "
            f"Load balance efficiency: {load_balance_efficiency_val:.2f}. "
            f"No KV head replication."
        ),
    }
