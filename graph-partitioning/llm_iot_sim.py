"""
llm_iot_sim.py
----------------
A lightweight simulator for distributing LLM inference across heterogeneous (wireless) IoT devices.

Supports policies:
  - "single_node": everything on one device.
  - "pipeline": layer-wise pipeline parallelism across devices.
  - "tensor": tensor parallelism across a group of devices with ring all-reduce model.
  - "metis": METIS k-way partitioning of the layer graph (if PyMetis is available),
    else falls back to a greedy balancer.

Model notes (TinyLlama-like by default):
  - 23 layers (including output head), 32 Q heads, 4 KV heads.
  - You can adjust model dimensions and coefficients for FLOPs and activation sizes to
    fit your needs.

Network model:
  - Simple symmetric wireless links (same RTT/bandwidth for all pairs) by default.
  - You can override a pairwise matrix if needed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Dict, Tuple, Optional

try:
    import pymetis  # Optional
    from pymetis import CSRAdjacency
except ImportError:
    pymetis = None
    CSRAdjacency = None


# ------------------------------
# Core data structures
# ------------------------------


class CommunicationModel(Enum):
    CENTRALIZED = "centralized"  # Client coordinates all communication
    PEER_TO_PEER = "peer_to_peer"  # Servers communicate directly


@dataclass
class ModelSpec:
    name: str = "TinyLlama-1.1B-Chat-v1.0"
    num_layers: int = 23  # includes output layer
    num_heads: int = 32
    num_kv_heads: int = 4
    d_model: int = 2048
    d_k: int = 64  # per-head
    d_ff: int = 5632
    communication_model: CommunicationModel = CommunicationModel.CENTRALIZED
    vocab_size: int = 32000

    # Coefficients for cost modeling (FLOPs per token per layer)
    # Very rough, tweak to calibrate against your measurements:
    attn_flops_coeff: float = 6.0  # scales ~ O(seq_len * d_model * num_heads)
    attn_quadratic_coeff: float = 1e-3  # scales ~ O(seq_len^2 * d_model) for qk matmul
    ffn_flops_coeff: float = 12.0  # scales ~ O(d_model * d_ff)
    # Activation transfer size per boundary (in bytes) per token:
    act_bytes_per_token: int = 4 * 2048  # float32 * d_model by default

    # Collective (tensor-parallel) all-reduce payload per layer & token (approx activations)
    allreduce_bytes_per_token: int = 4 * 2048

    # KV cache bytes per token per layer (read/written). Used for seq_len dependence in gen.
    kv_bytes_per_token_per_layer: int = (
        2 * 4 * 2048
    )  # (K,V) * float32 * d_model (very rough)

    def layer_compute_flops_per_token(self, seq_len: int) -> float:
        """
        Very rough per-layer compute FLOPs for generating one token given context length seq_len.
        """
        attn_linear = self.attn_flops_coeff * self.d_model * self.num_heads
        attn_quad = self.attn_quadratic_coeff * (seq_len**2) * self.d_model
        ffn = self.ffn_flops_coeff * self.d_model * self.d_ff
        return attn_linear + attn_quad + ffn


@dataclass
class Device:
    name: str
    gflops: float  # peak FP32 GFLOPs (or "effective" compute units)
    memory_gb: float
    net_bw_mbps: float  # throughput (Mbps) to the shared wireless fabric
    net_rtt_ms: float  # baseline RTT (ms) on the wireless fabric
    efficiency: float = 0.6  # how much of peak compute you realize during inference

    def flops_per_sec(self) -> float:
        return 1e9 * self.gflops * self.efficiency

    def compute_time_s(self, flops: float) -> float:
        return flops / self.flops_per_sec()


@dataclass
class Network:
    # Symmetric shared wireless fabric (simplified). If pairwise overrides provided,
    # they take precedence when computing link costs.
    default_bw_mbps: float = 300.0
    default_rtt_ms: float = 15.0
    pair_bw_mbps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pair_rtt_ms: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Communication model parameters for centralized vs peer-to-peer
    client_bandwidth_mbps: float = 1000.0  # Client-server bandwidth
    client_rtt_ms: float = 10.0  # Client-server RTT

    def link(self, a: str, b: str) -> Tuple[float, float]:
        if a == b:
            return float("inf"), 0.0
        key = (a, b)
        if key in self.pair_bw_mbps:  # directional override
            bw = self.pair_bw_mbps[key]
        elif (b, a) in self.pair_bw_mbps:
            bw = self.pair_bw_mbps[(b, a)]
        else:
            bw = self.default_bw_mbps

        if key in self.pair_rtt_ms:
            rtt = self.pair_rtt_ms[key]
        elif (b, a) in self.pair_rtt_ms:
            rtt = self.pair_rtt_ms[(b, a)]
        else:
            rtt = self.default_rtt_ms
        return bw, rtt

    def xfer_time_s(self, bytes_size: float, a: str, b: str) -> float:
        bw_mbps, rtt_ms = self.link(a, b)
        if math.isinf(bw_mbps):
            return 0.0
        return (bytes_size * 8.0 / (bw_mbps * 1e6)) + rtt_ms / 1000.0

    def ring_allreduce_time_with_pairwise_links(
        self, bytes_size: float, device_names: List[str]
    ) -> float:
        """
        Ring all-reduce time accounting for actual pairwise link characteristics.
        In a ring, each device sends to its next neighbor in the ring.
        Following this model:
            time ≈ 2 * (N - 1) / N * (bytes / bw) + (N - 1) * RTT
        """
        n = len(device_names)
        if n <= 1:
            return 0.0

        # Build ring: device[i] -> device[(i+1) % n]
        total_time = 0.0

        # Reduce-scatter phase: (n-1) steps, each step uses one ring link
        for _ in range(n - 1):
            # Find the bottleneck link for this step (all devices send simultaneously)
            step_time = 0.0
            for i in range(n):
                sender = device_names[i]
                receiver = device_names[(i + 1) % n]
                bw_mbps, rtt_ms = self.link(sender, receiver)

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = (chunk_size * 8.0 / (bw_mbps * 1e6)) + (rtt_ms / 1000.0)
                step_time = max(step_time, link_time)  # Bottleneck determines step time

            total_time += step_time

        # All-gather phase: (n-1) more steps
        for _ in range(n - 1):
            step_time = 0.0
            for i in range(n):
                sender = device_names[i]
                receiver = device_names[(i + 1) % n]
                bw_mbps, rtt_ms = self.link(sender, receiver)

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = (chunk_size * 8.0 / (bw_mbps * 1e6)) + (rtt_ms / 1000.0)
                step_time = max(step_time, link_time)

            total_time += step_time

        return total_time

    def compute_communication_cost(
        self,
        tensor_size_bytes: float,
        devices: List[Device],
        communication_model: CommunicationModel,
    ) -> float:
        """Compute communication cost based on the model type and actual pairwise links."""
        device_names = [d.name for d in devices]
        num_servers = len(devices)

        if communication_model == CommunicationModel.CENTRALIZED:
            # Actual RPC sequence: Graph Compute -> Get Tensor -> Add Data Compute Graph
            total_cost = 0.0

            # Phase 1: Client sends graph compute request to all servers (parallel)
            max_graph_compute_time = 0.0
            for _ in device_names:
                # Graph compute request (serialized subgraph + tensor descriptors)
                # Estimate ~1KB for small subgraph + tensor metadata per server
                request_size = 1024  # bytes for serialized graph
                request_time = (
                    request_size * 8.0 / (self.client_bandwidth_mbps * 1e6)
                ) + (self.client_rtt_ms / 1000.0)
                # Response is just status (small)
                response_time = self.client_rtt_ms / 1000.0
                graph_compute_time = request_time + response_time
                max_graph_compute_time = max(max_graph_compute_time, graph_compute_time)

            total_cost += max_graph_compute_time

            # Phase 2: Client requests partial tensor results from all servers (parallel)
            max_get_tensor_time = 0.0
            for _ in device_names:
                chunk_size = (
                    tensor_size_bytes / num_servers
                )  # Each server computed a chunk
                # Get tensor request (small tensor descriptor)
                request_time = self.client_rtt_ms / 1000.0
                # Response contains the actual tensor chunk data
                response_time = (
                    chunk_size * 8.0 / (self.client_bandwidth_mbps * 1e6)
                ) + (self.client_rtt_ms / 1000.0)
                get_tensor_time = request_time + response_time
                max_get_tensor_time = max(max_get_tensor_time, get_tensor_time)

            total_cost += max_get_tensor_time

            # Phase 3: Client sends aggregated tensor back to all servers
            # for local storage/reduction (parallel)
            max_allreduce_time = 0.0
            for _ in device_names:
                # All reduce compute graph request
                # (simple addition graph for local all-reduce + full aggregated tensor)
                graph_size = 512  # bytes for simple add op graph
                request_time = (
                    (graph_size + tensor_size_bytes)
                    * 8.0
                    / (self.client_bandwidth_mbps * 1e6)
                ) + (self.client_rtt_ms / 1000.0)
                # Response is just computation status
                response_time = self.client_rtt_ms / 1000.0
                allreduce_time = request_time + response_time
                max_allreduce_time = max(max_allreduce_time, allreduce_time)

            total_cost += max_allreduce_time

            # Add synchronization barrier overhead (thread joins, etc.)
            sync_overhead = 0.001 * num_servers  # 1ms per server for coordination
            total_cost += sync_overhead

            return total_cost

        # PEER_TO_PEER
        # Ring all-reduce using actual pairwise links
        return self.ring_allreduce_time_with_pairwise_links(
            tensor_size_bytes, device_names
        )


# ------------------------------
# Policy implementations
# ------------------------------


def simulate_single_node(
    device: Device, model: ModelSpec, seq_len: int, gen_tokens: int
) -> Dict:
    """All layers on one device; simple latency estimate."""
    per_layer_flops = model.layer_compute_flops_per_token(seq_len)
    per_token_flops = model.num_layers * per_layer_flops
    t_token = device.compute_time_s(per_token_flops)
    total = t_token * gen_tokens
    return {
        "policy": "single_node",
        "device": device.name,
        "per_token_s": t_token,
        "total_latency_s": total,
        "utilization_est": device.efficiency,
        "notes": "No comms; ignores prompt prefill time.",
    }


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
    devices: List[Device], model: ModelSpec, net: Network, seq_len: int, gen_tokens: int
) -> Dict:
    """
    Estimate pipeline-parallel token latency with simple bubble model.
    Assumes each boundary sends activation for each token once.
    """
    layer_costs = [
        model.layer_compute_flops_per_token(seq_len) for _ in range(model.num_layers)
    ]
    parts = _pipeline_partition_by_compute(devices, layer_costs)

    # Stage compute times per token
    stage_compute = []
    stage_pairs = []
    for stage_idx, layer_ids in enumerate(parts):
        stage_flops = sum(layer_costs[i] for i in layer_ids)
        t = devices[stage_idx].compute_time_s(stage_flops)
        stage_compute.append(t)
        if stage_idx > 0:
            a = devices[stage_idx - 1].name
            b = devices[stage_idx].name
            stage_pairs.append((a, b))
        else:
            stage_pairs.append(None)

    # Activation transfer times between stages per token
    act = model.act_bytes_per_token
    stage_xfer = []
    for pair in stage_pairs:
        if pair is None:
            stage_xfer.append(0.0)
        else:
            a, b = pair
            stage_xfer.append(net.xfer_time_s(act, a, b))

    # Per-token pipeline stage time (compute + inbound xfer)
    stage_times = [stage_compute[0]] + [
        stage_compute[i] + stage_xfer[i] for i in range(1, len(stage_compute))
    ]
    stage_time = max(stage_times)

    # Latency = warmup bubbles + steady-state
    bubbles = sum(stage_times) - stage_time  # naive bubble overhead per first token
    total = bubbles + stage_time * gen_tokens

    return {
        "policy": "pipeline",
        "stages": [
            {
                "device": d.name,
                "layers": parts[i],
                "compute_s": stage_compute[i],
                "inbound_xfer_s": stage_xfer[i],
            }
            for i, d in enumerate(devices)
        ],
        "per_token_s_steady": stage_time,
        "total_latency_s": total,
        "notes": "Simple pipeline with contiguous partitions and one activation send per boundary.",
    }


class AttentionSplitStrategy(Enum):
    KV_HEADS = "kv_heads"


class FFNSplitStrategy(Enum):
    KV_HEADS = "kv_heads"
    HIDDEN_DIM = "hidden_dim"
    Q_HEADS = "q_heads"


def simulate_tensor_parallel(
    devices: List[Device],
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    parallel_degree: Optional[int] = None,
    efficiency_loss: float = 0.85,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
) -> Dict:
    """
    Tensor-parallel over a group of devices with flexible head-level splitting strategies for attention and FFN.
    No KV head replication - limited by num_kv_heads.

    Supported combinations:
    1. attn: kv_heads, ffn: kv_heads     -> Both split by KV head groups
    2. attn: kv_heads, ffn: hidden_dim   -> Attention by KV heads, FFN by dimension
    3. attn: kv_heads, ffn: q_heads      -> Attention by KV heads, FFN by Q head groups.
    """
    if parallel_degree is None:
        k = len(devices)
    else:
        k = min(parallel_degree, len(devices))
        devices = devices[:k]

    # Attention is always split by KV heads (no replication)
    kv_heads_per_device = model.num_kv_heads // k
    remaining_kv = model.num_kv_heads % k

    device_kv_counts = [kv_heads_per_device] * min(k, model.num_kv_heads)
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

    # Calculate per-layer compute for each device
    per_layer_compute_times = []

    for i, device in enumerate(devices):
        device_q_heads = device_q_counts[i]
        device_kv_heads = device_kv_counts[i]
        device_ffn_fraction = device_ffn_fractions[i]

        # Attention FLOPs based on head allocation
        # Q projection scales with Q heads assigned to this device
        q_proj_flops = (
            model.attn_flops_coeff * model.d_model * device_q_heads * model.d_k
        )

        # K, V projections scale with KV heads assigned to this device
        kv_proj_flops = (
            2 * model.attn_flops_coeff * model.d_model * device_kv_heads * model.d_k
        )

        # Attention computation (QK^T, softmax, attention*V) scales with Q heads
        q_fraction = device_q_heads / model.num_heads
        attn_compute_flops = (
            model.attn_quadratic_coeff * (seq_len**2) * model.d_model * q_fraction
        )

        # Output projection scales with Q heads (determines output size)
        out_proj_flops = (
            model.attn_flops_coeff * model.d_model * device_q_heads * model.d_k
        )

        total_attn_flops = (
            q_proj_flops + kv_proj_flops + attn_compute_flops + out_proj_flops
        )

        # FFN FLOPs based on split strategy
        ffn_flops = (
            model.ffn_flops_coeff * model.d_model * model.d_ff * device_ffn_fraction
        )

        total_flops = total_attn_flops + ffn_flops
        compute_time = device.compute_time_s(total_flops) / efficiency_loss
        per_layer_compute_times.append(compute_time)

    # The bottleneck device determines the per-layer time (synchronous execution)
    per_layer_time = max(per_layer_compute_times)
    compute_time = per_layer_time * model.num_layers

    # All-reduce cost per layer (assume 2 collectives per layer)
    red_bytes = model.allreduce_bytes_per_token

    # Attention all-reduce: always needed after attention computation
    attn_devices = [
        device
        for device, attn_frac in zip(devices, device_attn_fractions)
        if attn_frac > 0
    ]
    attn_comm_cost = net.compute_communication_cost(
        red_bytes, attn_devices, model.communication_model
    )

    # FFN all-reduce: needed after FFN computation
    ffn_devices = [
        device
        for device, ffn_frac in zip(devices, device_ffn_fractions)
        if ffn_frac > 0
    ]
    ffn_comm_cost = net.compute_communication_cost(
        red_bytes, ffn_devices, model.communication_model
    )

    # Total communication per layer: attention + FFN collectives
    per_layer_comm = attn_comm_cost + ffn_comm_cost
    comm_time = per_layer_comm * model.num_layers

    per_token = compute_time + comm_time
    total = per_token * gen_tokens

    # Calculate load balance and memory metrics
    compute_time_std = (
        sum((t - per_layer_time) ** 2 for t in per_layer_compute_times) / k
    ) ** 0.5
    load_balance_efficiency = min(per_layer_compute_times) / max(
        per_layer_compute_times
    )

    # Memory usage per device (KV cache + model weights)
    device_memory_usage = []
    for i in range(k):
        # KV cache memory for assigned KV heads
        kv_cache_bytes = (
            device_kv_counts[i] * model.d_k * 2 * 4 * seq_len
        )  # (K,V) * d_k * float32 * seq_len

        # Model weights (rough estimate)
        # Attention weights: Q + K + V + output projections
        attn_weights = (
            device_q_counts[i] * model.d_k * model.d_model * 4  # Q proj
            + device_kv_counts[i] * model.d_k * model.d_model * 2 * 4  # K,V proj
        )

        # FFN weights based on split strategy
        ffn_weights = (
            device_ffn_fractions[i] * model.d_model * model.d_ff * 2 * 4
        )  # up + down proj

        total_memory = (
            kv_cache_bytes + (attn_weights + ffn_weights) * model.num_layers
        ) / (
            1024**3
        )  # GB
        device_memory_usage.append(total_memory)

    return {
        "policy": "tensor",
        "communication_model": model.communication_model.value,
        "k": k,
        "split_strategies": {
            "attention": attn_split_strategy,
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
        "component_times_s": {
            "compute": compute_time,
            "communication": comm_time,
            "attn_communication": attn_comm_cost * model.num_layers,
            "ffn_communication": ffn_comm_cost * model.num_layers,
        },
        "efficiency_metrics": {
            "load_balance_efficiency": load_balance_efficiency,
            "compute_time_std": compute_time_std,
        },
        "total_latency_s": total,
        "notes": (
            f"Using {model.communication_model.value} communication with pairwise link modeling. "
            f"Attention split by {attn_split_strategy}, FFN split by {ffn_split_strategy}. "
            f"Load balance efficiency: {load_balance_efficiency:.2f}. "
            f"No KV head replication."
        ),
    }


def metis_partition_layers(k: int, model: ModelSpec, seq_len: int) -> List[List[int]]:
    """
    Build a chain graph of layers with node weights ~ compute and edge weights ~ activation size.
    Return contiguous or non-contiguous partitions (METIS can split arbitrarily).
    If PyMetis is not available, fall back to a greedy balancer by compute.
    """
    if pymetis is None or CSRAdjacency is None:
        # Fallback: contiguous greedy like pipeline
        layer_costs = [
            model.layer_compute_flops_per_token(seq_len)
            for _ in range(model.num_layers)
        ]
        # Create k buckets with similar total compute
        parts = [[] for _ in range(k)]
        sums = [0.0] * k
        for i, c in enumerate(layer_costs):
            idx = sums.index(min(sums))
            parts[idx].append(i)
            sums[idx] += c
        return parts

    # METIS expects adjacency list; create a simple chain graph 0-1-2-...-(L-1)
    n = model.num_layers
    xadj = [0]
    adjncy = []
    eweights = []
    for i in range(n):
        nbrs = []
        wts = []
        if i - 1 >= 0:
            nbrs.append(i - 1)
            wts.append(1)  # edge weight placeholder
        if i + 1 < n:
            nbrs.append(i + 1)
            wts.append(1)
        adjncy.extend(nbrs)
        eweights.extend(wts)
        xadj.append(len(adjncy))

    # Node weights (ints) ~ compute cost
    nodewgt = [
        max(1, int(1e-6 * model.layer_compute_flops_per_token(seq_len)))
        for _ in range(n)
    ]

    csr = CSRAdjacency(
        adj_starts=xadj,
        adjacent=adjncy,
    )
    _, assignment = pymetis.part_graph(
        k, adjacency=csr, vweights=nodewgt, eweights=eweights
    )
    parts: List[List[int]] = [[] for _ in range(k)]
    for i, p in enumerate(assignment):
        parts[p].append(i)
    return parts


def simulate_metis(
    devices: List[Device], model: ModelSpec, net: Network, seq_len: int, gen_tokens: int
) -> Dict:
    """
    Partition layers with METIS (or greedy) and schedule as a generalized pipeline.
    We assume a linear order of parts by average layer index to approximate a pipeline path.
    """
    k = len(devices)
    parts = metis_partition_layers(k, model, seq_len)

    # Order parts by average layer index to form a pipeline order
    order = sorted(range(k), key=lambda p: sum(parts[p]) / max(1, len(parts[p])))
    parts_ordered = [parts[i] for i in order]
    devices_ordered = [devices[i] for i in order]

    # Reuse pipeline estimation but allow non-contiguous sets per stage by summing costs
    layer_costs = [
        model.layer_compute_flops_per_token(seq_len) for _ in range(model.num_layers)
    ]
    stage_compute = []
    for idx, stage_layers in enumerate(parts_ordered):
        flops = sum(layer_costs[i] for i in stage_layers)
        stage_compute.append(devices_ordered[idx].compute_time_s(flops))

    # Assume one activation send between consecutive stages per token
    act = model.act_bytes_per_token
    stage_xfer = [0.0]
    for i in range(1, k):
        a = devices_ordered[i - 1].name
        b = devices_ordered[i].name
        stage_xfer.append(net.xfer_time_s(act, a, b))

    stage_times = [stage_compute[0]] + [
        stage_compute[i] + stage_xfer[i] for i in range(1, k)
    ]
    stage_time = max(stage_times)
    bubbles = sum(stage_times) - stage_time
    total = bubbles + stage_time * gen_tokens

    return {
        "policy": "metis",
        "stages": [
            {
                "device": devices_ordered[i].name,
                "layers": parts_ordered[i],
                "compute_s": stage_compute[i],
                "inbound_xfer_s": stage_xfer[i],
            }
            for i in range(k)
        ],
        "per_token_s_steady": stage_time,
        "total_latency_s": total,
        "notes": (
            "Order by average layer index to form a pipeline; "
            "METIS fallback is greedy if PyMetis unavailable."
        ),
    }


# ------------------------------
# Experiment helper
# ------------------------------


def compare_tensor_parallel_strategies(
    devices: List[Device],
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    parallel_degree: Optional[int] = None,
) -> Dict[str, Dict]:
    """Compare the three tensor parallel splitting strategies."""

    strategies = [
        (
            AttentionSplitStrategy.KV_HEADS,
            FFNSplitStrategy.KV_HEADS,
        ),  # Both split by KV head groups - most aligned
        (
            AttentionSplitStrategy.KV_HEADS,
            FFNSplitStrategy.HIDDEN_DIM,
        ),  # Attention by KV heads, FFN traditional split
        (
            AttentionSplitStrategy.KV_HEADS,
            FFNSplitStrategy.Q_HEADS,
        ),  # Attention by KV heads, FFN by Q head groups
    ]

    results = {}
    for attn_strategy, ffn_strategy in strategies:
        strategy_name = f"attn_{attn_strategy.name}_ffn_{ffn_strategy.name}"
        results[strategy_name] = simulate_tensor_parallel(
            devices,
            model,
            net,
            seq_len,
            gen_tokens,
            parallel_degree=parallel_degree,
            attn_split_strategy=attn_strategy,
            ffn_split_strategy=ffn_strategy,
        )

    return results


def run_experiments(
    devices: List[Device],
    model: Optional[ModelSpec] = None,
    net: Optional[Network] = None,
    seq_len: int = 512,
    gen_tokens: int = 64,
) -> Dict[str, Dict]:
    """
    Run all policies on the given device set.
    Returns a dict of results keyed by policy.
    """
    model = model or ModelSpec()
    net = net or Network()

    results = {}
    # Single node (fastest device)
    fastest = max(devices, key=lambda d: d.gflops * d.efficiency)
    results["single_node"] = simulate_single_node(fastest, model, seq_len, gen_tokens)

    # Pipeline (use all devices)
    results["pipeline"] = simulate_pipeline(devices, model, net, seq_len, gen_tokens)

    # Tensor (use all devices)
    results["tensor"] = simulate_tensor_parallel(
        devices, model, net, seq_len, gen_tokens
    )

    # METIS
    results["metis"] = simulate_metis(devices, model, net, seq_len, gen_tokens)
    return results


if __name__ == "__main__":
    # Example: 4 heterogeneous Raspberry Pi-like devices
    SERVERS = [
        Device(
            "rpi-1",
            gflops=12.5,
            memory_gb=4.0,
            net_bw_mbps=300,
            net_rtt_ms=12,
            efficiency=0.55,
        ),
        Device(
            "rpi-2",
            gflops=10.0,
            memory_gb=2.0,
            net_bw_mbps=250,
            net_rtt_ms=15,
            efficiency=0.5,
        ),
        Device(
            "rpi-3",
            gflops=7.5,
            memory_gb=2.0,
            net_bw_mbps=200,
            net_rtt_ms=18,
            efficiency=0.5,
        ),
        Device(
            "rpi-4",
            gflops=5.0,
            memory_gb=1.0,
            net_bw_mbps=150,
            net_rtt_ms=20,
            efficiency=0.45,
        ),
    ]

    # Shared wireless fabric
    NET = Network(default_bw_mbps=200.0, default_rtt_ms=18.0)

    # Model + workload
    MODEL = ModelSpec()
    SEQ_LEN = 512
    GEN_TOKENS = 64

    res = run_experiments(SERVERS, MODEL, NET, SEQ_LEN, GEN_TOKENS)
    from pprint import pprint

    pprint(res)
