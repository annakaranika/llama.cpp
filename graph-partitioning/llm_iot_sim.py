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
import logging
import math
from pprint import pformat
from typing import List, Dict, Tuple, Optional

try:
    import pymetis  # Optional
    from pymetis import CSRAdjacency
except ImportError:
    pymetis = None
    CSRAdjacency = None

# ------------------------------
# Logging setup
# ------------------------------


def setup_logger(
    name: str = "llm_iot_sim",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logger for LLM IoT simulation.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create module-level logger
LOGGER = setup_logger()


# ------------------------------
# Core data structures
# ------------------------------


class CommunicationModel(Enum):
    """Communication model for distributed inference."""

    CENTRALIZED = "centralized"  # Client coordinates all communication
    PEER_TO_PEER = "peer_to_peer"  # Servers communicate directly


class AttentionSplitStrategy(Enum):
    """How to split attention heads across devices."""

    KV_HEADS = "kv_heads"


class FFNSplitStrategy(Enum):
    """How to split FFN layers across devices."""

    KV_HEADS = "kv_heads"
    HIDDEN_DIM = "hidden_dim"
    Q_HEADS = "q_heads"


@dataclass
class Device:
    """
    Represents a compute device in the distributed inference simulation.
    Attributes:
        name (str): Device identifier.
        gflops (float): Peak FP32 GFLOPs (or "effective" compute units).
        memory_gb (float): Available memory in GB.

    """

    name: str
    gflops: float  # effective/achievable GFLOPs (or "effective" compute units)
    memory_gb: float

    def flops_per_sec(self) -> float:
        """Effective FLOPs/s."""
        return 1e9 * self.gflops

    def compute_time_s(self, flops: float) -> float:
        """Compute time in seconds for given FLOPs."""
        return flops / self.flops_per_sec()


@dataclass
class ModelSpec:
    """
    Model specification class for LLM inference simulation.

    This class defines the architectural parameters and cost modeling coefficients
    for a language model, specifically configured for TinyLlama-1.1B-Chat-v1.0.
    It includes parameters for transformer architecture, communication patterns,
    and performance estimation coefficients.

    Attributes:
        name (str): Model identifier, defaults to "TinyLlama-1.1B-Chat-v1.0"
        num_layers (int): Total number of transformer layers including output layer
        num_heads (int): Number of attention heads in multi-head attention
        num_kv_heads (int): Number of key-value heads for grouped query attention
        d_model (int): Model dimensionality/hidden size
        d_k (int): Dimension per attention head
        d_ff (int): Feed-forward network hidden dimension
        communication_model (CommunicationModel): Communication pattern for distributed inference
        vocab_size (int): Size of the model's vocabulary
        attn_flops_coeff (float): Coefficient for attention linear operations FLOPs estimation
        attn_quadratic_coeff (float): Coefficient for attention quadratic complexity FLOPs
        ffn_flops_coeff (float): Coefficient for feed-forward network FLOPs estimation
        act_bytes_per_token (int): Activation transfer size in bytes per token at layer boundaries
        allreduce_bytes_per_token (int): Collective communication payload per token per layer
        kv_bytes_per_token_per_layer (int): KV cache memory usage per token per layer

    Methods:
        layer_compute_flops_per_token(seq_len): Estimates FLOPs per token per layer
            given sequence length, accounting for both linear and quadratic attention costs.

    Note:
        Coefficients are rough estimates and should be calibrated against actual measurements
        for accurate performance modeling.
    """

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

    def attn_compute_flops(
        self,
        seq_len: int,
        dev_num_heads: Optional[int] = None,
        dev_kv_heads: Optional[int] = None,
    ) -> float:
        """
        Compute attention compute flops given sequence length and heads.
        """
        device_q_heads = dev_num_heads if dev_num_heads is not None else self.num_heads
        device_kv_heads = (
            dev_kv_heads if dev_kv_heads is not None else self.num_kv_heads
        )

        # Attention FLOPs based on head allocation
        # Q projection scales with Q heads assigned to this device
        q_proj_flops = self.attn_flops_coeff * self.d_model * device_q_heads * self.d_k

        # K, V projections scale with KV heads assigned to this device
        kv_proj_flops = (
            2 * self.attn_flops_coeff * self.d_model * device_kv_heads * self.d_k
        )

        # Attention computation (QK^T, softmax, attention*V) scales with Q heads
        q_fraction = device_q_heads / self.num_heads
        attn_compute_flops = (
            self.attn_quadratic_coeff * (seq_len**2) * self.d_model * q_fraction
        )

        # Output projection scales with Q heads (determines output size)
        out_proj_flops = (
            self.attn_flops_coeff * self.d_model * device_q_heads * self.d_k
        )

        total_attn_flops = (
            q_proj_flops + kv_proj_flops + attn_compute_flops + out_proj_flops
        )
        return total_attn_flops

    def ffn_compute_flops(self, ffn_fraction: float = 1.0) -> float:
        """
        Compute FFN compute flops given FFN fraction.
        """
        ffn_flops = self.ffn_flops_coeff * self.d_model * self.d_ff * ffn_fraction
        return ffn_flops

    def layer_compute_flops(
        self,
        seq_len: int,
        dev_num_heads: Optional[int] = None,
        dev_kv_heads: Optional[int] = None,
        ffn_fraction: float = 1.0,
    ) -> float:
        """
        Very rough per-layer compute FLOPs for generating one token given context length seq_len.
        """
        attn = self.attn_compute_flops(
            seq_len,
            dev_num_heads=dev_num_heads,
            dev_kv_heads=dev_kv_heads,
        )
        ffn = self.ffn_compute_flops(ffn_fraction=ffn_fraction)
        return attn + ffn

    def layer_compute_cost_per_device(
        self,
        seq_len: int,
        devices: List[Device],
        device_q_counts: Optional[List[int]] = None,
        device_kv_counts: Optional[List[int]] = None,
        device_ffn_fractions: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute per-layer compute times across a list of devices
        given their attention and FFN allocations.
        Returns the total compute time (s) for the layer across all devices,
        as well as compute time standard deviation and load balance efficiency.
        """

        per_layer_compute_times = []

        for i, device in enumerate(devices):
            device_q_heads = device_q_counts[i] if device_q_counts else self.num_heads
            device_kv_heads = (
                device_kv_counts[i] if device_kv_counts else self.num_kv_heads
            )
            device_ffn_fraction = (
                device_ffn_fractions[i] if device_ffn_fractions else 1.0
            )

            attn_flops = self.attn_compute_flops(
                seq_len,
                dev_num_heads=device_q_heads,
                dev_kv_heads=device_kv_heads,
            )

            ffn_flops = self.ffn_compute_flops(ffn_fraction=device_ffn_fraction)

            total_flops = attn_flops + ffn_flops
            compute_time = device.compute_time_s(total_flops)
            per_layer_compute_times.append(compute_time)

        # The bottleneck device determines the per-layer time (synchronous execution)
        per_layer_time = max(per_layer_compute_times)
        compute_time = per_layer_time * self.num_layers

        return per_layer_compute_times

    def memory_usage_per_device(
        self,
        seq_len: int,
        devices: List[Device],
        device_kv_counts: Optional[List[int]] = None,
        device_ffn_fractions: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Estimate memory usage (GB) per device given KV head counts and FFN fractions.
        """
        memory_usages = []
        for i in range(len(devices)):
            # KV cache memory for assigned KV heads
            kv_heads = device_kv_counts[i] if device_kv_counts else self.num_kv_heads
            kv_cache_bytes = (
                kv_heads * self.d_k * 2 * 4 * seq_len
            )  # (K,V) * d_k * float32 * seq_len

            # Model weights (rough estimate)
            # Attention weights: Q + K + V + output projections
            q_heads = device_kv_counts[i] if device_kv_counts else self.num_heads
            attn_weights = (
                q_heads * self.d_k * self.d_model * 4  # Q proj
                + kv_heads * self.d_k * self.d_model * 2 * 4  # K,V proj
            )

            # FFN weights based on split strategy
            ffn_fraction = device_ffn_fractions[i] if device_ffn_fractions else 1.0
            ffn_weights = (
                ffn_fraction * self.d_model * self.d_ff * 2 * 4
            )  # up + down proj

            total_memory = (
                kv_cache_bytes + (attn_weights + ffn_weights) * self.num_layers
            ) / (
                1024**3
            )  # GB
            memory_usages.append(total_memory)
        return memory_usages


@dataclass
class Network:
    """
    Represents a network with symmetric shared wireless fabric.
    If pairwise overrides are provided, they take precedence when computing link costs.
    """

    client: Optional[Device] = None
    servers: List[Device] = field(default_factory=list)
    default_bw_mbps: float = 300.0
    default_rtt_ms: float = 15.0
    pair_bw_mbps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pair_rtt_ms: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Communication model parameters for centralized vs peer-to-peer
    client_bandwidth_mbps: float = 1000.0  # Client-server bandwidth
    client_rtt_ms: float = 10.0  # Client-server RTT

    def __post_init__(self):
        if self.client is None:
            self.client = Device("client", gflops=15.0, memory_gb=4.0)

    def link(self, a: str, b: str) -> Tuple[float, float]:
        """
        Get bandwidth (Mbps) and RTT (ms) between two devices.
        """
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
        """
        Estimate transfer time between two devices considering bandwidth and RTT.
        """
        bw_mbps, rtt_ms = self.link(a, b)
        if math.isinf(bw_mbps):
            return 0.0
        return (bytes_size * 8.0 / (bw_mbps * 1e6)) + rtt_ms / 1000.0

    def avg_client_server_time(self, bytes_size: float) -> float:
        """
        Client-server transfer time model.
        """
        bw_rtt_pairs = [self.link("client", server.name) for server in self.servers]
        if bw_rtt_pairs:
            avg_bw = sum(bw for bw, _ in bw_rtt_pairs) / len(bw_rtt_pairs)
            avg_rtt = sum(rtt for _, rtt in bw_rtt_pairs) / len(bw_rtt_pairs)
        else:
            avg_bw = self.client_bandwidth_mbps
            avg_rtt = self.client_rtt_ms

        return (bytes_size * 8.0 / (avg_bw * 1e6)) + (avg_rtt / 1000.0)

    def centralized_allreduce_time(self, bytes_size: float, num_servers: int) -> float:
        """
        Centralized all-reduce time model.
        Assumes client coordinates all communication.
        """
        if num_servers <= 1:
            return 0.0
        # Each server sends and receives the full data once
        time = 2 * (bytes_size * 8.0 / (self.client_bandwidth_mbps * 1e6)) + (
            2 * self.client_rtt_ms / 1000.0
        )
        return time

    def ring_allreduce_time_with_pairwise_links(
        self, bytes_size: float, device_names: List[str]
    ) -> float:
        """
        Ring all-reduce time accounting for actual pairwise link characteristics.
        In a ring, each device sends to its next neighbor in the ring.
        Following this model:
            time ≈ 2 * (N - 1) / N * (bytes / bw) + (N - 1) * RTT

        TODO: take into consideration the case that splits are not the same number
        between attention and FFN blocks.
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

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = self.xfer_time_s(chunk_size, sender, receiver)
                step_time = max(step_time, link_time)  # Bottleneck determines step time

            total_time += step_time

        # All-gather phase: (n-1) more steps
        for _ in range(n - 1):
            step_time = 0.0
            for i in range(n):
                sender = device_names[i]
                receiver = device_names[(i + 1) % n]

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = self.xfer_time_s(chunk_size, sender, receiver)
                step_time = max(step_time, link_time)

            total_time += step_time

        return total_time

    def allreduce_communication_cost(
        self,
        tensor_size_bytes: float,
        devices: List[Device],
        communication_model: CommunicationModel,
    ) -> float:
        """Compute communication cost based on the model type and actual pairwise links."""
        num_servers = len(devices)

        if communication_model == CommunicationModel.CENTRALIZED:
            # Actual RPC sequence: Graph Compute -> Get Tensor -> Add Data Compute Graph
            total_cost = 0.0

            # Phase 1: Client sends graph compute request to all servers (parallel)
            max_graph_compute_time = 0.0
            for _ in devices:
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
            for _ in devices:
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
            for _ in devices:
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
            tensor_size_bytes, device_names=[d.name for d in devices]
        )


# ------------------------------
# Metrics and utilities
# ------------------------------


def std_times(times: List[float]) -> float:
    """
    Compute standard deviation of compute times.
    """
    n = len(times)
    if n == 0:
        return 0.0
    mean = sum(times) / n
    variance = sum((t - mean) ** 2 for t in times) / n
    return math.sqrt(variance)


def efficiency(times: List[float]) -> float:
    """
    Compute load balance efficiency as mean / max compute time.
    """
    if not times:
        return 0.0
    mean = sum(times) / len(times)
    max_time = max(times)
    if max_time == 0:
        return 0.0
    return mean / max_time


# ------------------------------
# Policy implementations
# ------------------------------


def simulate_single_node(
    model: ModelSpec, net: Network, seq_len: int, gen_tokens: int
) -> Dict:
    """All layers on one device; simple latency estimate."""
    devices = net.servers
    if not devices:
        raise ValueError("No devices available in the network.")
    fastest = max(net.servers, key=lambda d: d.gflops)
    per_layer_flops = model.layer_compute_flops(seq_len)
    per_token_flops = model.num_layers * per_layer_flops
    t_token = fastest.compute_time_s(per_token_flops)
    total = t_token * gen_tokens
    return {
        "policy": "single_node",
        "device": fastest.name,
        "per_token_s": t_token,
        "total_latency_s": total,
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
    model: ModelSpec, net: Network, seq_len: int, gen_tokens: int
) -> Dict:
    """
    Estimate pipeline-parallel token latency with simple bubble model.
    Assumes each boundary sends activation for each token once.
    """
    devices = net.servers
    layer_costs = [model.layer_compute_flops(seq_len) for _ in range(model.num_layers)]
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


def simulate_tensor_parallel(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    parallel_degree: Optional[int] = None,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
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
    k = (
        len(net.servers)
        if parallel_degree is None
        else min(parallel_degree, len(net.servers))
    )
    devices = net.servers[:k] if parallel_degree is not None else net.servers

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
    per_layer_compute_times = model.layer_compute_cost_per_device(
        seq_len,
        devices,
        device_q_counts,
        device_kv_counts,
        device_ffn_fractions,
    )
    compute_time = max(per_layer_compute_times) * model.num_layers

    # All-reduce cost per layer (assume 2 collectives per layer)
    red_bytes = model.allreduce_bytes_per_token

    # Attention all-reduce: always needed after attention computation
    attn_devices = [
        device
        for device, attn_frac in zip(devices, device_attn_fractions)
        if attn_frac > 0
    ]
    attn_comm_cost = net.allreduce_communication_cost(
        red_bytes, attn_devices, model.communication_model
    )

    # FFN all-reduce: needed after FFN computation
    ffn_devices = [
        device
        for device, ffn_frac in zip(devices, device_ffn_fractions)
        if ffn_frac > 0
    ]
    ffn_comm_cost = net.allreduce_communication_cost(
        red_bytes, ffn_devices, model.communication_model
    )

    # Total communication per layer: attention + FFN collectives
    per_layer_comm = attn_comm_cost + ffn_comm_cost
    comm_time = per_layer_comm * model.num_layers

    per_token = compute_time + comm_time
    total = per_token * gen_tokens

    # Calculate load balance and memory metrics
    compute_time_std = std_times(per_layer_compute_times)
    load_balance_efficiency_val = efficiency(per_layer_compute_times)

    # Memory usage per device (KV cache + model weights)
    device_memory_usage = model.memory_usage_per_device(
        seq_len, devices, device_kv_counts, device_ffn_fractions
    )

    return {
        "policy": "tensor",
        "communication_model": model.communication_model.value,
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
        "notes": (
            f"Using {model.communication_model.value} communication with pairwise link modeling. "
            f"Attention split by {attn_split_strategy}, FFN split by {ffn_split_strategy}. "
            f"Load balance efficiency: {load_balance_efficiency_val:.2f}. "
            f"No KV head replication."
        ),
    }


def build_computation_graph(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Build a computation graph with separate attention and FFN blocks per layer.

    Graph structure per layer:
    - Attention block: parallel nodes (one per possible split)
    - FFN block: parallel nodes (one per possible split)
    - Intra-block edges: communication cost between parallel attention/FFN nodes
    - Inter-block edges: single edge from attention[0] -> ffn[0] (activation transfer)

    Returns:
    - xadj: Adjacency list starts
    - adjncy: Adjacent nodes
    - eweights: Edge weights
    - node_types: Node type indicators (0=attn, 1=ffn)
    - node_splits: Which split each node represents
    - node_weights: Node weights based on computation flops
    """

    LOGGER.info("Building computation graph for %s", model.name)
    LOGGER.info(
        "Strategies: attention=%s, ffn=%s",
        attn_split_strategy.name,
        ffn_split_strategy.name,
    )

    k = len(net.servers)
    n_layers = model.num_layers

    # Calculate split possibilities
    max_attn_splits = (
        model.num_kv_heads
        if attn_split_strategy == AttentionSplitStrategy.KV_HEADS
        else 1
    )
    max_ffn_splits = {
        FFNSplitStrategy.KV_HEADS: model.num_kv_heads,
        FFNSplitStrategy.Q_HEADS: model.num_heads,
        FFNSplitStrategy.HIDDEN_DIM: k,  # Can split arbitrarily across hidden dim
    }[ffn_split_strategy]

    LOGGER.debug("Max splits: attention=%s, ffn=%s", max_attn_splits, max_ffn_splits)

    # Node layout: [
    #   layer_0_attn_0, layer_0_attn_1, ..., layer_0_ffn_0, layer_0_ffn_1, ..., layer_1_attn_0, ...
    # ]
    nodes_per_layer = max_attn_splits + max_ffn_splits
    total_nodes = n_layers * nodes_per_layer

    LOGGER.info(
        "Graph structure: %s total nodes, %s nodes per layer",
        total_nodes,
        nodes_per_layer,
    )

    xadj = [0]  # Start of adjacency for each node
    adjncy = []  # Adjacent nodes
    eweights = []  # Edge weights
    node_types = []  # 0=attention, 1=ffn
    split_ids = []  # Split ID for each node (1-indexed within block type)
    node_weights = []  # Node weights based on computation cost

    def get_node_id(layer: int, block_type: int, split_id: int) -> int:
        """Get global node ID for layer/block_type/split_id"""
        return layer * nodes_per_layer + block_type * max_attn_splits + split_id

    # Build adjacency for each node
    for node_id in range(total_nodes):
        layer = node_id // nodes_per_layer
        within_layer = node_id % nodes_per_layer

        if within_layer < max_attn_splits:
            # Attention node
            block_type = 0
            split_id = within_layer
            max_block_splits = max_attn_splits

            # Compute attention cost for this split degree
            device_kv_heads = model.num_kv_heads // max_attn_splits
            device_q_heads = (model.num_heads // model.num_kv_heads) * device_kv_heads

            compute_flops = model.attn_compute_flops(
                seq_len, dev_num_heads=device_q_heads, dev_kv_heads=device_kv_heads
            )

        else:
            # FFN node
            block_type = 1
            split_id = within_layer - max_attn_splits
            max_block_splits = max_ffn_splits

            # Compute FFN cost for this split degree
            ffn_fraction = 1.0 / max_ffn_splits
            compute_flops = model.ffn_compute_flops(ffn_fraction=ffn_fraction)

        node_weight = max(1, int(compute_flops))

        split_ids.append(split_id)
        node_types.append(block_type)
        node_weights.append(node_weight)

        nbrs = []  # Neighbor nodes
        wts = []  # Corresponding edge weights

        # 1. Intra-block edges (parallel nodes in same block incur extra communication cost)
        if split_id == 0:
            for other_split_id in range(max_block_splits):
                if other_split_id != split_id:
                    other_node = get_node_id(layer, block_type, other_split_id)
                    nbrs.append(other_node)

                    # Communication cost between nodes (tensor parallel all-reduce)
                    comm_cost = net.allreduce_communication_cost(
                        model.allreduce_bytes_per_token,
                        net.servers,
                        model.communication_model,
                    )
                    edge_weight = max(1, int(comm_cost * 1000))
                    wts.append(edge_weight)
        else:
            # Connect back to first split node
            other_node = get_node_id(layer, block_type, 0)
            nbrs.append(other_node)

            comm_cost = net.allreduce_communication_cost(
                model.allreduce_bytes_per_token,
                net.servers,
                model.communication_model,
            )
            edge_weight = max(1, int(comm_cost * 1000))
            wts.append(edge_weight)

        if (  # 2. Inter-block edge (attention -> FFN within same layer)
            block_type == 0 and split_id == 0
        ):  # Only first attention node connects to first FFN node
            next_block_node = get_node_id(layer, 1, 0)
            if layer > 0:
                # Also connect from previous layer's FFN to this layer's attention
                prev_block_node = get_node_id(layer - 1, 1, 0)
            else:
                prev_block_node = None

        elif (  # 3. Inter-layer edges (FFN -> next layer attention)
            block_type == 1 and split_id == 0
        ):  # Only first FFN node connects to next layer
            if layer < n_layers - 1:
                next_block_node = get_node_id(layer + 1, 0, 0)
            else:
                next_block_node = None
            prev_block_node = get_node_id(layer, 0, 0)  # From this layer's attention

        else:  # other block nodes
            next_block_node = None
            prev_block_node = None

        if next_block_node is not None:
            nbrs.append(next_block_node)

            comm_cost = net.allreduce_communication_cost(
                model.allreduce_bytes_per_token,
                net.servers[: max(max_attn_splits, max_ffn_splits)],
                model.communication_model,
            )
            edge_weight = max(1, int(comm_cost * 1000))
            wts.append(edge_weight)

        if prev_block_node is not None:
            nbrs.append(prev_block_node)

            comm_cost = net.allreduce_communication_cost(
                model.allreduce_bytes_per_token,
                net.servers[: max(max_attn_splits, max_ffn_splits)],
                model.communication_model,
            )
            edge_weight = max(1, int(comm_cost * 1000))
            wts.append(edge_weight)

        adjncy.extend(nbrs)
        eweights.extend(wts)
        xadj.append(len(adjncy))

    def get_node_name(node_id: int) -> str:
        """Get human-readable node name"""
        layer = node_id // nodes_per_layer
        within_layer = node_id % nodes_per_layer
        if within_layer < max_attn_splits:
            block_type = "attn"
            split_id = within_layer
        else:
            block_type = "ffn"
            split_id = within_layer - max_attn_splits
        return f"layer_{layer}_{block_type}_{split_id}"

    # Print first few nodes to show structure
    LOGGER.debug("Building node adjacencies...")
    for i in range(min(20, total_nodes)):
        layer = i // nodes_per_layer
        within_layer = i % nodes_per_layer
        block_type = "attn" if within_layer < max_attn_splits else "ffn"
        split_id = (
            within_layer
            if within_layer < max_attn_splits
            else within_layer - max_attn_splits
        )

        neighbors = zip(
            [get_node_name(node_id) for node_id in adjncy[xadj[i] : xadj[i + 1]]],
            eweights[xadj[i] : xadj[i + 1]],
        )

        LOGGER.debug(
            "Node %s: %s -> compute: %s, neighbors: %s",
            i,
            get_node_name(i),
            node_weights[i],
            list(neighbors),
        )

    if total_nodes > 20:
        LOGGER.debug("... and %s more nodes", total_nodes - 20)

    LOGGER.info(
        "Graph construction complete: %s edges, %s nodes",
        len(adjncy),
        len(node_weights),
    )

    return xadj, adjncy, eweights, node_types, split_ids, node_weights


def device_constraints_for_metis(servers: List[Device]) -> List[float]:
    """
    Compute device constraints for METIS.
    Higher weight = device can handle more computation.
    """
    capacities = [s.gflops for s in servers]
    # Set target partition weights based on device capacities
    total_capacity = sum(capacities)
    weights = [w / total_capacity for w in capacities]

    return weights


def metis_partition_computation_blocks(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
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
        build_computation_graph(
            model, net, seq_len, attn_split_strategy, ffn_split_strategy
        )
    )

    # Compute device capacity weights
    target_partition_weights = device_constraints_for_metis(net.servers)
    LOGGER.info(
        "Device capacity weights: %s",
        [
            f"{net.servers[i].name}={w:.3f}"
            for i, w in enumerate(target_partition_weights)
        ],
    )

    # Run METIS partitioning
    LOGGER.info("Running METIS graph partitioning...")

    csr = CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
    _, assignment = pymetis.part_graph(
        k,
        adjacency=csr,
        vweights=node_weights,
        eweights=edge_weights,
        tpwgts=target_partition_weights,
    )
    LOGGER.info("METIS partitioning completed successfully")

    # Analyze the partitioning results
    partitions = [[] for _ in range(k)]
    for node_id, partition_id in enumerate(assignment):
        layer = node_id // (len(node_types) // model.num_layers)
        block_type = node_types[node_id]
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

    return {
        "partitions": partitions,
        "total_nodes": len(node_types),
        "graph_structure": {
            "attention_blocks": sum(1 for t in node_types if t == 0),
            "ffn_blocks": sum(1 for t in node_types if t == 1),
        },
        "assignment": assignment,
    }


def simulate_metis(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
) -> Dict:
    """
    Enhanced METIS simulation with attention/FFN block structure.
    """
    partition_result = metis_partition_computation_blocks(
        model, net, seq_len, attn_split_strategy, ffn_split_strategy
    )

    devices = net.servers
    partitions = partition_result["partitions"]

    LOGGER.debug("Computing device workloads from partition assignments...")

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
        actual_flops = total_flops  # Already in the right units from graph construction
        compute_time = device.compute_time_s(actual_flops)

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

    # Per-layer timing: max across devices (synchronous execution)
    per_layer_compute_time = max(device_compute_times) if device_compute_times else 0.0
    total_compute_time = per_layer_compute_time * model.num_layers

    # Communication happens per layer as well
    per_layer_comm_time = communication_costs["per_layer_total"]
    total_comm_time = per_layer_comm_time * model.num_layers

    # Total per-token time
    per_token_time = total_compute_time + total_comm_time
    total_latency = per_token_time * gen_tokens

    # Load balance analysis
    compute_time_std = std_times(device_compute_times)
    load_balance_efficiency_val = efficiency(device_compute_times)

    # Memory analysis - estimate based on assigned blocks
    memory_usage = estimate_metis_memory_usage(partitions, model, seq_len)

    return {
        "policy": "metis",
        # "partition_info": partition_result,
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
        "notes": (
            f"METIS partitioning with {len(partitions)} devices. "
            f"Load balance efficiency: {load_balance_efficiency_val:.3f}. "
            f"Communication pattern derived from graph structure."
        ),
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

    # Build partition assignment mapping: node_id -> device_id
    node_to_device = {}
    node_names = {}
    for device_idx, partition_nodes in enumerate(partitions):
        for node in partition_nodes:
            node_to_device[node["node_id"]] = device_idx
            node_names[node["node_id"]] = node["node_name"]

    # Rebuild the graph to analyze cross-partition communication
    xadj, adjncy, edge_weights, node_types, _, node_weights = build_computation_graph(
        model, net, seq_len
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

            edge_info = {
                "source_node": node_names[node_id],
                "target_node": node_names[neighbor_id],
                "source_device": source_device,
                "target_device": neighbor_device,
                "weight": edge_weight,
                "source_type": "attn" if node_types[node_id] == 0 else "ffn",
                "target_type": "attn" if node_types[neighbor_id] == 0 else "ffn",
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
        actual_comm_cost = edge["weight"] / 1000.0  # We scaled by 1000
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
    partitions: List[List[Dict]], model: ModelSpec, seq_len: int
) -> List[float]:
    """
    Estimate memory usage per device based on METIS partition assignments.
    """
    memory_usage = []

    for partition_nodes in partitions:
        if not partition_nodes:
            memory_usage.append(0.0)
            continue

        # Count attention and FFN blocks assigned to this device
        attention_blocks = sum(
            1 for node in partition_nodes if node["block_type"] == "attn"
        )
        ffn_blocks = sum(1 for node in partition_nodes if node["block_type"] == "ffn")

        # Estimate model weights memory
        # Rough approximation: each attention/FFN block needs model parameters
        layers_involved = len(set(node["layer"] for node in partition_nodes))

        # Model parameters (very rough estimate)
        params_per_attn_block = (
            model.d_model * model.d_k * model.num_heads * 4
        )  # Q,K,V,O projections
        params_per_ffn_block = model.d_model * model.d_ff * 2  # up + down projections

        total_params = (
            attention_blocks * params_per_attn_block + ffn_blocks * params_per_ffn_block
        )
        param_memory_gb = total_params * 4 / (1024**3)  # FP32

        # KV cache memory (scales with layers involved and sequence length)
        kv_cache_gb = (
            layers_involved * seq_len * model.num_kv_heads * model.d_k * 2 * 4
        ) / (1024**3)

        # Activation memory (rough estimate)
        activation_gb = layers_involved * seq_len * model.d_model * 4 / (1024**3)

        total_memory = param_memory_gb + kv_cache_gb + activation_gb
        memory_usage.append(total_memory)

    return memory_usage


# ------------------------------
# Experiment helper
# ------------------------------


def compare_tensor_parallel_strategies(
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
    model: Optional[ModelSpec] = None,
    net: Optional[Network] = None,
    seq_len: int = 512,
    gen_tokens: int = 64,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run all policies on the given device set.
    Returns a dict of results keyed by policy.
    """
    # Setup logger for this experiment run
    exp_logger = setup_logger(level=log_level, log_file=log_file)

    exp_logger.info("=" * 60)
    exp_logger.info("Starting LLM IoT simulation experiments")
    exp_logger.info("=" * 60)

    model = model or ModelSpec()
    net = net or Network()

    exp_logger.info("Model: %s (%s layers)", model.name, model.num_layers)
    exp_logger.info("Devices: %s", [d.name + f"({d.gflops}GF)" for d in net.servers])
    exp_logger.info("Workload: %s context + %s generation tokens", seq_len, gen_tokens)

    results = {}
    # Single node (fastest device)
    results["single_node"] = simulate_single_node(model, net, seq_len, gen_tokens)

    # Pipeline (use all devices)
    results["pipeline"] = simulate_pipeline(model, net, seq_len, gen_tokens)

    # Tensor (use all devices)
    results["tensor"] = simulate_tensor_parallel(model, net, seq_len, gen_tokens)

    # METIS
    results["metis"] = simulate_metis(model, net, seq_len, gen_tokens)

    LOGGER.info("Experiment results:\n%s", pformat(results, width=100, depth=3))

    return results


if __name__ == "__main__":
    # Example: 4 heterogeneous Raspberry Pi-like devices
    CLIENT = Device("client", gflops=100.0, memory_gb=16.0)
    SERVERS = [
        Device("rpi-1", gflops=12.5, memory_gb=4.0),
        Device("rpi-2", gflops=10.0, memory_gb=2.0),
        Device("rpi-3", gflops=5.5, memory_gb=2.0),
        Device("rpi-4", gflops=0.4, memory_gb=1.0),
    ]

    # Shared wireless fabric
    NET = Network(CLIENT, SERVERS, default_bw_mbps=200.0, default_rtt_ms=18.0)

    # Model + workload
    MODEL = ModelSpec()
    SEQ_LEN = 512
    GEN_TOKENS = 64

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"llm_iot_sim_{timestamp}.log"
    # res = simulate_metis(
    #     MODEL, NET, SEQ_LEN, GEN_TOKENS  # , ffn_split_strategy=FFNSplitStrategy.Q_HEADS
    # )
    res = run_experiments(MODEL, NET, SEQ_LEN, GEN_TOKENS, log_file=LOG_FILE)
