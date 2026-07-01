"""METIS-based hybrid-parallelism graph-partitioning policy for single-request inference."""

from typing import Dict, List, Optional

try:
    import pymetis
    from pymetis import CSRAdjacency
except ImportError:
    pymetis = None
    CSRAdjacency = None

from ..common import (
    AttentionSplitStrategy,
    FFNSplitStrategy,
    MetisDevConsideration,
    std_times,
    efficiency,
)
from ..graph_plots import (
    draw_partitions,
    draw_device_computation_breakdown,
    plot_resource_utilization,
)
import os
import time

from ..logging_utils import setup_logger, setup_output_dir
from ..model import ModelSpec
from ..network import Network

os.makedirs("sim_output", exist_ok=True)
LOGGER = setup_logger(name=__name__, log_file="sim_output/metis.log")


class MetisCtx:
    """Context for METIS partitioning."""

    def __init__(self):
        self.output_dir = ""
        # Accumulated wall-clock spent in diagnostic plotting, so it can be
        # subtracted from the reported planning time (plotting is a simulator
        # artifact, not part of computing the allocation).
        self.plot_seconds = 0.0

    def setup_output_dir(self, parent_dir: str):
        """Set up output directory for METIS results."""
        self.output_dir = setup_output_dir(parent_dir)

    def get_output_dir(self) -> str:
        """Get the output directory for METIS results."""
        return self.output_dir

    def reset_plot_timer(self):
        """Zero the accumulated plotting time (call at the start of a run)."""
        self.plot_seconds = 0.0

    def timed_plot(self, fn, *args, **kwargs):
        """Run a plotting call, accumulating its duration into plot_seconds."""
        _t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.plot_seconds += time.perf_counter() - _t


CTX = MetisCtx()


def compute_safe_partition_weights(
    mem_capacities: List[float],
    gflop_capacities: List[float],
    total_memory_needed: float,
) -> List[float]:
    """
    Computes partition weights proportional to gflop capacities,
    but strictly capped by memory capacity ratios to prevent swapping.
    """
    n = len(mem_capacities)
    # Memory cap for each device as a fraction of total memory needed
    mem_caps = [mem / total_memory_needed for mem in mem_capacities]
    
    # Check if model fits in total memory of devices with non-zero gflops
    active_indices = [i for i in range(n) if gflop_capacities[i] > 0.0]
    total_active_mem = sum(mem_capacities[i] for i in active_indices)
    
    if total_active_mem < total_memory_needed:
        # If it doesn't fit in active devices, distribute weights proportional
        # to memory capacities of active devices to minimize swap/overhead.
        if total_active_mem > 0:
            return [mem_capacities[i] / total_active_mem if i in active_indices else 0.0 for i in range(n)]
        else:
            return [1.0 / len(active_indices) if i in active_indices else 0.0 for i in range(n)]

    # Initialize weights proportional to gflops for active devices
    weights = [gflop_capacities[i] for i in range(n)]
    total_gflops = sum(weights)
    if total_gflops > 0:
        weights = [w / total_gflops for w in weights]
    else:
        weights = [1.0 / len(active_indices) if i in active_indices else 0.0 for i in range(n)]
        
    # Iterative capping and redistribution
    capped_indices = set()
    for _ in range(n):
        excess = 0.0
        for i in active_indices:
            if i not in capped_indices and weights[i] > mem_caps[i]:
                excess += weights[i] - mem_caps[i]
                weights[i] = mem_caps[i]
                capped_indices.add(i)
                
        if excess < 1e-9:
            break
            
        # Redistribute excess to uncapped active indices
        uncapped_indices = [i for i in active_indices if i not in capped_indices]
        uncapped_gflops = sum(gflop_capacities[i] for i in uncapped_indices)
        
        if uncapped_gflops > 0:
            for i in uncapped_indices:
                weights[i] += excess * (gflop_capacities[i] / uncapped_gflops)
        else:
            if uncapped_indices:
                share = excess / len(uncapped_indices)
                for i in uncapped_indices:
                    weights[i] += share
            else:
                break
                
    # Normalize final weights
    total_w = sum(weights)
    if total_w > 0:
        weights = [w / total_w for w in weights]
    return weights



def device_constraints_for_metis(
    model: ModelSpec,
    net: Network,
    dev_consideration: MetisDevConsideration = MetisDevConsideration.CONSTRAINING_RESOURCE,
    mem_seq_len: int = 512,
) -> List[float]:
    """
    Device constraints for METIS.
    Considers compute (GFLOPS), memory (GB), and network (Mbps) capacities.
    ``mem_seq_len`` is the KV-cache length to use for memory pressure.
    """
    mem_capacities = [s.memory_gb for s in net.servers]
    gflop_capacities = [s.gflops for s in net.servers]
    net_capacities = [net.aggregate_bandwidth_capacity(s) for s in net.servers]
    server_capacities = {}
    per_layer_memory = model.memory_usage_per_device(
        mem_seq_len, num_layers=[1]
    )[0]
    considered_resource = []

    def assigned_resource(chosen, mem, comp, netw) -> str:
        if chosen == mem:
            return "Memory"
        elif chosen == comp:
            return "Compute"
        elif chosen == netw:
            return "Network"
        else:
            return "Unknown"

    for i in range(len(net.servers)):  # pylint:disable=consider-using-enumerate
        if mem_capacities[i] < per_layer_memory:
            server_capacities[i] = 0.0
            continue
        # mem_cap_ratio = mem_capacities[i] / sum(mem_capacities)
        mem_cap_ratio = mem_capacities[i] / model.total_memory_usage(seq_len=mem_seq_len)
        gflop_cap_ratio = gflop_capacities[i] / sum(gflop_capacities)
        net_cap_ratio = net_capacities[i] / sum(net_capacities)
        if dev_consideration == MetisDevConsideration.MEMORY_ONLY:
            server_capacities[i] = mem_cap_ratio
        elif dev_consideration == MetisDevConsideration.COMPUTE_ONLY:
            server_capacities[i] = gflop_cap_ratio
        elif dev_consideration == MetisDevConsideration.NETWORK_ONLY:
            server_capacities[i] = net_cap_ratio
        elif dev_consideration == MetisDevConsideration.CONSTRAINING_RESOURCE:
            server_capacities[i] = min(mem_cap_ratio, gflop_cap_ratio, net_cap_ratio)
        elif dev_consideration == MetisDevConsideration.DOMINANT_RESOURCE:
            server_capacities[i] = max(mem_cap_ratio, gflop_cap_ratio, net_cap_ratio)
        else:
            raise ValueError(f"Unknown MetisDevConsideration: {dev_consideration}")
        chosen = server_capacities[i]
        considered_resource.append(
            assigned_resource(chosen, mem_cap_ratio, gflop_cap_ratio, net_cap_ratio)
        )
        LOGGER.info(
            "Device %s: mem_cap_ratio=%.3f, gflop_cap_ratio=%.3f, net_cap_ratio=%.3f, chosen=%.3f",
            net.servers[i].repr(net.aggregate_bandwidth_capacity(net.servers[i])),
            mem_cap_ratio,
            gflop_cap_ratio,
            net_cap_ratio,
            server_capacities[i],
        )

    sorted_capacities = sorted(
        server_capacities.items(), key=lambda x: x[1], reverse=True
    )
    needed_mem = model.memory_usage_per_device(
        mem_seq_len, num_layers=[model.num_layers]
    )[0]
    selected_weights = {idx: 0.0 for idx, _ in sorted_capacities}
    selected_mem = 0.0
    for idx, cap in sorted_capacities:
        if selected_mem >= needed_mem:
            break
        if cap == 0.0:
            continue
        selected_weights[idx] = 1.0
        selected_mem += mem_capacities[idx]

    # Selected devices have selected_weights[idx] == 1.0
    selected_gflops = [
        gflop_capacities[i] if selected_weights.get(i, 0.0) > 0.0 else 0.0
        for i in range(len(net.servers))
    ]

    assignments = compute_safe_partition_weights(
        mem_capacities=mem_capacities,
        gflop_capacities=selected_gflops,
        total_memory_needed=needed_mem,
    )

    LOGGER.info(
        "Totals: %s",
        [
            [
                sum(gflop_capacities),
                model.total_memory_usage(seq_len=mem_seq_len),
                sum(net_capacities),
            ]
        ],
    )

    CTX.setup_output_dir("sim_output")
    CTX.timed_plot(
        plot_resource_utilization,
        ["Compute", "Memory", "Network"],
        ["GFLOPs", "GB", "Mbps"],
        [
            sum(gflop_capacities),
            model.total_memory_usage(seq_len=mem_seq_len),
            sum(net_capacities),
        ],
        net.servers,
        considered_resource,
        assignments,
        save_path=f"{CTX.get_output_dir()}/metis_resource_utilization.pdf",
    )

    LOGGER.info("Device shares: %s", assignments)

    return assignments


def metis_partition_computation_blocks(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    dev_consideration: MetisDevConsideration = MetisDevConsideration.CONSTRAINING_RESOURCE,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """
    Build a chain graph of layers with node weights ~ compute and edge weights ~ communication cost
    based on the model's communication strategy.
    Return partitions optimized for the specified communication model.
    If PyMetis is not available, fall back to a greedy balancer by compute.
    """

    LOGGER.info("Starting METIS partitioning")

    k = len(net.servers)
    LOGGER.info("Partitioning into %s parts for %s devices", k, len(net.servers))

    if pymetis is None or CSRAdjacency is None:
        # Fallback: contiguous greedy like pipeline
        layer_costs = [
            model.layer_compute_flops(seq_len) for _ in range(model.num_layers)
        ]
        # Create k buckets with similar total compute
        parts = [[] for _ in range(k)]
        sums = [0.0] * k
        for i, c in enumerate(layer_costs):
            idx = sums.index(min(sums))
            parts[idx].append(i)
            sums[idx] += c
        return {"partitions": parts}

    # Build computation graph
    LOGGER.debug("Building computation graph...")
    xadj, adjncy, edge_weights, node_types, split_ids, node_weights = (
        model.build_computation_graph(
            net, seq_len, attn_split_strategy, ffn_split_strategy
        )
    )

    # Compute device capacity weights
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    target_partition_weights = device_constraints_for_metis(
        model, net, dev_consideration, mem_seq_len=mem_seq_len
    )
    LOGGER.info(
        "Device capacity weights: %s",
        [
            f"{net.servers[i].name}={w:.3f}"
            for i, w in enumerate(target_partition_weights)
        ],
    )

    # Run METIS partitioning. pymetis rejects tpwgts arrays with zero entries
    # ("Incorrect tpwgts for partition X and constraint 0"), so we drop
    # zero-weight devices, partition over the survivors, and remap the
    # assignment back to the original device index space.
    kept_indices = [
        i for i, w in enumerate(target_partition_weights) if w > 0.0
    ]
    if not kept_indices:
        raise RuntimeError(
            "METIS: no device has positive partition weight; the model "
            "does not fit on any configured device."
        )

    reduced_weights = [target_partition_weights[i] for i in kept_indices]
    total = sum(reduced_weights)
    reduced_weights = [w / total for w in reduced_weights]
    # Pin floating-point drift so pymetis sees an exact sum of 1.0.
    reduced_weights[-1] = 1.0 - sum(reduced_weights[:-1])
    reduced_k = len(kept_indices)

    LOGGER.info(
        "Running METIS with %d/%d devices (kept indices: %s)",
        reduced_k,
        k,
        kept_indices,
    )

    if reduced_k == 1:
        # METIS can't partition into a single part; assign everything there.
        assignment = [kept_indices[0]] * len(node_weights)
    else:
        csr = CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
        _, reduced_assignment = pymetis.part_graph(
            reduced_k,
            adjacency=csr,
            vweights=node_weights,
            eweights=edge_weights,
            tpwgts=reduced_weights,
        )
        assignment = [kept_indices[p] for p in reduced_assignment]
    LOGGER.info("METIS partitioning completed successfully")

    # Analyze the partitioning results. Skip virtual all-reduce nodes
    # (node_types 2 and 3); they exist only to carry collective-comm cost
    # in the cut and do not represent compute that must be assigned.
    partitions = [[] for _ in range(k)]
    nodes_per_layer = len(node_types) // model.num_layers
    for node_id, partition_id in enumerate(assignment):
        block_type = node_types[node_id]
        if block_type not in (0, 1):
            continue  # virtual AR node — has no compute payload
        layer = node_id // nodes_per_layer
        split_id = split_ids[node_id]

        partitions[partition_id].append(
            {
                "node_id": node_id,
                "node_name": f"layer_{layer}_{'attn' if block_type == 0 else 'ffn'}_{split_id}",
                "layer": layer,
                "block_type": "attn" if block_type == 0 else "ffn",
                "split_id": split_id,
                "compute_weight": node_weights[node_id],
            }
        )

    max_attn_splits = model.get_max_attn_splits(attn_split_strategy)
    max_ffn_splits = model.get_max_ffn_splits(ffn_split_strategy, len(net.servers))
    dev_net_aggr = (
        net.dev_net_aggregation_strategy.name
        if net.dev_net_aggregation_strategy
        else "None"
    )
    peer_to_peer_policy = net.peer2peer_policy.name if net.peer2peer_policy else "None"

    # create file with setup info for the output directory
    with open(f"{CTX.get_output_dir()}/setup_info.yaml", "w", encoding="utf-8") as f:
        f.write("policy: metis\n")
        f.write(f"model: {model.name}\n")
        f.write(f"num_layers: {model.num_layers}\n")
        f.write(f"attention_split_strategy: {attn_split_strategy.name}\n")
        f.write(f"ffn_split_strategy: {ffn_split_strategy.name}\n")
        f.write(f"max_attn_splits: {max_attn_splits}\n")
        f.write(f"max_ffn_splits: {max_ffn_splits}\n")
        f.write(f"graph_node_num: {len(node_weights)}\n")
        f.write(f"devices: {[d.name for d in net.servers]}\n")
        f.write(f"communication_model: {net.communication_model.name}\n")
        f.write(f"peer_to_peer_policy: {peer_to_peer_policy}\n")
        f.write(f"dev_consideration: {dev_consideration.name}\n")
        f.write(f"dev_net_aggregation: {dev_net_aggr}\n")
        f.write(f"partition_weights: {target_partition_weights}\n")

    CTX.timed_plot(
        draw_partitions,
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
        save_path=f"{CTX.get_output_dir()}/metis_partition_graph.pdf",
        policy_name="METIS",
        full_assignment=list(assignment),
    )

    CTX.timed_plot(
        draw_device_computation_breakdown,
        partitions,
        net.servers,
        save_path=f"{CTX.get_output_dir()}/metis_device_computation_breakdown.pdf",
    )

    return {
        "partitions": partitions,
        "total_nodes": len(node_types),
        "graph_structure": {
            "attention_blocks": sum(1 for t in node_types if t == 0),
            "ffn_blocks": sum(1 for t in node_types if t == 1),
        },
        "assignment": assignment,
    }


def analyze_metis_communication_costs(
    partition_result: Dict,
    model: ModelSpec,
    net: Network,
    seq_len: int,
) -> Dict:
    """
    Analyze communication costs from METIS partition by examining
    cross-partition edges in the computation graph.
    """
    partitions = partition_result["partitions"]

    # Build partition assignment mapping: node_id -> device_id.
    # ``partitions`` only contains compute-bearing split nodes, but virtual
    # all-reduce nodes also have METIS assignments and we need them here so
    # split->virtual edges get classified correctly. Fall back to the raw
    # ``assignment`` list (returned by metis_partition_computation_blocks)
    # for any node that isn't in the split-only partitions dict.
    node_to_device = {}
    node_names = {}
    for device_idx, partition_nodes in enumerate(partitions):
        for node in partition_nodes:
            node_to_device[node["node_id"]] = device_idx
            node_names[node["node_id"]] = node["node_name"]
    full_assignment = partition_result.get("assignment", [])
    for node_id, device_idx in enumerate(full_assignment):
        node_to_device.setdefault(node_id, int(device_idx))

    # Rebuild the graph to analyze cross-partition communication
    xadj, adjncy, edge_weights, node_types, _, node_weights = (
        model.build_computation_graph(net, seq_len)
    )

    # Identify cross-partition edges (communication requirements)
    intra_partition_edges = []
    cross_partition_edges = []

    for node_id in range(len(node_weights)):
        if node_id not in node_to_device:
            continue

        source_device = node_to_device[node_id]

        # Check all neighbors of this node
        start_idx = xadj[node_id]
        end_idx = xadj[node_id + 1]

        for adj_idx in range(start_idx, end_idx):
            neighbor_id = adjncy[adj_idx]
            edge_weight = edge_weights[adj_idx]

            if neighbor_id not in node_to_device:
                continue

            neighbor_device = node_to_device[neighbor_id]

            _TYPE_NAME = {0: "attn", 1: "ffn", 2: "attn_AR", 3: "ffn_AR"}
            edge_info = {
                "source_node": node_names.get(node_id, f"node_{node_id}"),
                "target_node": node_names.get(neighbor_id, f"node_{neighbor_id}"),
                "source_device": source_device,
                "target_device": neighbor_device,
                "weight": edge_weight,
                "source_type": _TYPE_NAME.get(node_types[node_id], "unknown"),
                "target_type": _TYPE_NAME.get(node_types[neighbor_id], "unknown"),
            }

            if source_device == neighbor_device:
                intra_partition_edges.append(edge_info)
            else:
                cross_partition_edges.append(edge_info)

    # Compute communication costs
    total_cross_partition_cost = 0.0
    device_pair_costs = {}

    for edge in cross_partition_edges:
        # Convert METIS edge weight back to actual communication cost
        # (reverse the scaling from build_computation_graph)
        actual_comm_cost = edge["weight"] / 1000000.0  # We scaled by 1000000.0
        total_cross_partition_cost += actual_comm_cost

        # Track costs between device pairs
        device_pair = (edge["source_device"], edge["target_device"])
        if device_pair not in device_pair_costs:
            device_pair_costs[device_pair] = 0.0
        device_pair_costs[device_pair] += actual_comm_cost

    return {
        "cross_partition_edges": len(cross_partition_edges),
        "intra_partition_edges": len(intra_partition_edges),
        "per_layer_total": total_cross_partition_cost,
        "device_pair_costs": device_pair_costs,
        "communication_pattern": {
            "attention_to_ffn": len(
                [
                    e
                    for e in cross_partition_edges
                    if e["source_type"] == "attn" and e["target_type"] == "ffn"
                ]
            ),
            "ffn_to_attention": len(
                [
                    e
                    for e in cross_partition_edges
                    if e["source_type"] == "ffn" and e["target_type"] == "attn"
                ]
            ),
            "attention_to_attention": len(
                [
                    e
                    for e in cross_partition_edges
                    if e["source_type"] == "attn" and e["target_type"] == "attn"
                ]
            ),
            "ffn_to_ffn": len(
                [
                    e
                    for e in cross_partition_edges
                    if e["source_type"] == "ffn" and e["target_type"] == "ffn"
                ]
            ),
        },
        "edge_details": cross_partition_edges[:10],  # First 10 for debugging
    }


def estimate_metis_memory_usage(
    partitions: List[List[Dict]],
    model: ModelSpec,
    seq_len: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    num_devices: int = 1,
) -> List[float]:
    """
    Estimate memory usage per device based on METIS partition assignments,
    consistent with ModelSpec's memory calculation.
    """
    memory_usage = []
    
    max_attn_splits = model.get_max_attn_splits(attn_split_strategy)
    max_ffn_splits = model.get_max_ffn_splits(ffn_split_strategy, num_devices)
    
    attn_mem_per_split = (model.attn_memory_gb_per_kv_head(seq_len) * model.num_kv_heads) / max_attn_splits
    ffn_mem_per_split = model.ffn_memory_gb_per_fraction(1.0) / max_ffn_splits

    for partition_nodes in partitions:
        if not partition_nodes:
            memory_usage.append(0.0)
            continue

        # Count attention and FFN blocks assigned to this device
        attention_blocks = sum(
            1 for node in partition_nodes if node["block_type"] == "attn"
        )
        ffn_blocks = sum(1 for node in partition_nodes if node["block_type"] == "ffn")

        total_memory = attention_blocks * attn_mem_per_split + ffn_blocks * ffn_mem_per_split
        memory_usage.append(total_memory)

    return memory_usage


def simulate_metis(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    dev_consideration: MetisDevConsideration = MetisDevConsideration.CONSTRAINING_RESOURCE,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict:
    """
    Enhanced METIS simulation with attention/FFN block structure.
    """
    # Planning time spans the METIS partitioning and cost analysis. The
    # partitioner emits diagnostic plots inline; those are timed via
    # CTX.timed_plot and subtracted below so planning_time_s reflects only the
    # cost of computing the allocation (consistent with the other policies).
    CTX.reset_plot_timer()
    _t0 = time.perf_counter()

    partition_result = metis_partition_computation_blocks(
        model,
        net,
        seq_len,
        attn_split_strategy,
        ffn_split_strategy,
        dev_consideration,
        kv_cache_seq_len=kv_cache_seq_len,
    )

    devices = net.servers
    ref_gflops = sum(s.gflops for s in devices) / len(devices) if devices else 1.0
    partitions = partition_result["partitions"]

    LOGGER.debug("Computing device workloads from partition assignments...")

    # Memory analysis - estimate based on assigned blocks
    mem_seq_len = kv_cache_seq_len if kv_cache_seq_len is not None else seq_len
    memory_usage = estimate_metis_memory_usage(
        partitions,
        model,
        mem_seq_len,
        attn_split_strategy,
        ffn_split_strategy,
        len(devices),
    )

    # Compute per-device workload from actual partition assignments
    device_compute_times = []
    device_workloads = []

    for device_idx, partition_nodes in enumerate(partitions):
        if not partition_nodes:
            device_compute_times.append(0.0)
            device_workloads.append(
                {
                    "total_flops": 0,
                    "attention_blocks": 0,
                    "ffn_blocks": 0,
                    "layers_involved": set(),
                }
            )
            continue

        device = devices[device_idx]
        total_flops = 0
        attention_blocks = 0
        ffn_blocks = 0
        layers_involved = set()

        # Sum up computation from assigned nodes
        for node in partition_nodes:
            total_flops += node["compute_weight"]
            layers_involved.add(node["layer"])

            if node["block_type"] == "attn":
                attention_blocks += 1
            else:
                ffn_blocks += 1

        # Convert METIS weights back to actual FLOPs (reverse the scaling)
        actual_flops = total_flops * (ref_gflops * 1000.0)
        compute_time = device.compute_time_s(
            actual_flops, memory_usage[device_idx], seq_len=seq_len
        )

        device_compute_times.append(compute_time)
        device_workloads.append(
            {
                "total_flops": actual_flops,
                "attention_blocks": attention_blocks,
                "ffn_blocks": ffn_blocks,
                "layers_involved": layers_involved,
                "nodes_assigned": len(partition_nodes),
            }
        )

    # Compute communication costs from graph structure
    LOGGER.debug("Analyzing communication costs...")
    communication_costs = analyze_metis_communication_costs(
        partition_result, model, net, seq_len
    )

    LOGGER.info(
        "Communication analysis: %s cross-partition edges",
        communication_costs["cross_partition_edges"],
    )

    # Compute times layer-by-layer — COMPUTE ONLY (swap is added once, below).
    # Within a layer the partition's devices run in parallel (max); layers run
    # in series (sum). Each (layer, device) goes through compute_time_s so metis
    # uses the same roofline (compute vs weight-streaming bandwidth) and seq-aware
    # R_compute curve as every other policy — not a bare flops/gflops, which
    # dropped the bandwidth-bound decode wall. The device's total streamed bytes
    # (memory_usage[dev_idx]) are apportioned to its layers in proportion to their
    # FLOPs so the per-layer bandwidth terms sum back to the device's full stream;
    # memory=0 keeps the DRAM-overflow swap penalty a once-per-forward charge below.
    dev_total_flops = [w["total_flops"] for w in device_workloads]
    layer_compute_times = []
    for layer in range(model.num_layers):
        layer_dev_times = []
        for dev_idx, dev in enumerate(devices):
            # Find nodes in this partition that belong to this layer
            layer_nodes = [node for node in partitions[dev_idx] if node["layer"] == layer]
            layer_flops = sum(node["compute_weight"] for node in layer_nodes) * (ref_gflops * 1000.0)
            frac = layer_flops / dev_total_flops[dev_idx] if dev_total_flops[dev_idx] else 0.0
            layer_mem = memory_usage[dev_idx] * frac
            layer_dev_times.append(
                dev.compute_time_s(
                    layer_flops, 0.0, bytes_streamed_gb=layer_mem, seq_len=seq_len
                )
            )
        layer_compute_times.append(max(layer_dev_times) if layer_dev_times else 0.0)

    # DRAM-overflow paging is a ONE-TIME cost per forward pass, not per layer.
    # The old code passed each device's *total* memory into the per-layer
    # compute_time_s above, so an overflowing device was charged the full paging
    # penalty on every layer (~num_layers× too much). Charge it once instead, on
    # the slowest overflowing device (devices page their own overflow in
    # parallel). compute_time_s with 0 FLOPs returns exactly the paging penalty,
    # reusing the Device swap model. Matches simulate_tensor_parallel.
    swap_penalty_s = max(
        (dev.compute_time_s(0.0, memory_usage[dev_idx])
         for dev_idx, dev in enumerate(devices)),
        default=0.0,
    )

    total_compute_time = sum(layer_compute_times) + swap_penalty_s
    per_layer_compute_time = total_compute_time / model.num_layers if model.num_layers > 0 else 0.0

    # Communication is analyzed from all cross-partition edges in the whole graph (already total)
    total_comm_time = communication_costs["per_layer_total"]
    per_layer_comm_time = total_comm_time / model.num_layers if model.num_layers > 0 else 0.0

    # Coordinator injection/extraction (input -> entry device, logits <- exit
    # device), CLIENT_SERVER only. Entry/exit approximated by the first/last
    # non-empty partition's device.
    occupied = [devices[i].name for i, p in enumerate(partitions) if p]
    coord_io = net.coordinator_io_time_s(
        model.activation_bytes * seq_len,
        model.activation_bytes,  # per-token result egress (small), not full logits
        occupied[0] if occupied else devices[0].name,
        occupied[-1] if occupied else devices[-1].name,
    )

    # lm_head / output embedding streamed once per token on the exit device, so
    # all policies stream the full model (single_node/pipeline already do).
    occupied_idx = [i for i, p in enumerate(partitions) if p]
    exit_dev = devices[occupied_idx[-1]] if occupied_idx else devices[-1]
    lm_head_s = exit_dev.compute_time_s(0.0, model.embedding_weights_gb())
    # Total per-token time
    per_token_time = total_compute_time + total_comm_time + coord_io + lm_head_s
    total_latency = per_token_time * gen_tokens

    # Load balance analysis
    compute_time_std = std_times(device_compute_times)
    load_balance_efficiency_val = efficiency(device_compute_times)
    # Subtract the inline diagnostic plotting so this is just the allocation cost.
    planning_time_s = (time.perf_counter() - _t0) - CTX.plot_seconds

    with open(f"{CTX.get_output_dir()}/setup_info.yaml", "a", encoding="utf-8") as f:
        # f.write(f"load_balance_efficiency: {load_balance_efficiency_val:.3f}\n")
        # f.write(f"compute_time_std: {compute_time_std:.6f}\n")
        f.write(f"total_latency_s: {total_latency:.3f}\n")

    return {
        "policy": "metis",
        "partitions": partitions,
        "device_workloads": {
            devices[i].name: {
                **device_workloads[i],
                "compute_time_s": device_compute_times[i],
                "memory_usage_gb": memory_usage[i],
            }
            for i in range(len(devices))
        },
        "timing_analysis": {
            "per_layer_compute_s": per_layer_compute_time,
            "per_layer_communication_s": per_layer_comm_time,
            "total_compute_s": total_compute_time,
            "total_communication_s": total_comm_time,
            "per_token_s": per_token_time,
        },
        "communication_breakdown": communication_costs,
        "efficiency_metrics": {
            "load_balance_efficiency": load_balance_efficiency_val,
            "compute_time_std": compute_time_std,
            "devices_utilized": sum(
                1 for w in device_workloads if w["total_flops"] > 0
            ),
        },
        "total_latency_s": total_latency,
        "planning_time_s": planning_time_s,
        "notes": (
            f"METIS partitioning with {len(partitions)} devices. "
            f"Load balance efficiency: {load_balance_efficiency_val:.3f}. "
            f"Communication pattern derived from graph structure."
        ),
    }
