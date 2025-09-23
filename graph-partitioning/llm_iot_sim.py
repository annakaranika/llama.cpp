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
from typing import List, Dict, Tuple, Optional
import math

try:
    import pymetis  # Optional
    from pymetis import CSRAdjacency
except ImportError:
    pymetis = None
    CSRAdjacency = None


# ------------------------------
# Core data structures
# ------------------------------


@dataclass
class ModelSpec:
    name: str = "TinyLlama-1.1B-Chat-v1.0"
    num_layers: int = 23  # includes output layer
    num_heads: int = 32
    num_kv_heads: int = 4
    d_model: int = 2048
    d_k: int = 64  # per-head
    d_ff: int = 5632
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


def ring_allreduce_time_s(
    bytes_size: float, devices: List[Device], net: Network
) -> float:
    """
    Time for a ring all-reduce over a homogeneous link assumption, approximated as:
      time ≈ 2 * (N - 1) / N * (bytes / bw) + (N - 1) * RTT
    We estimate bw as the min pairwise bw to be pessimistic.
    """
    n = len(devices)
    if n <= 1:
        return 0.0
    # Conservative: use worst-case link between any two participants
    min_bw = float("inf")
    max_rtt = 0.0
    names = [d.name for d in devices]
    for i in range(n):
        for j in range(i + 1, n):
            bw, rtt = net.link(names[i], names[j])
            min_bw = min(min_bw, bw)
            max_rtt = max(max_rtt, rtt)
    data_time = 2 * (n - 1) / n * (bytes_size * 8.0 / (min_bw * 1e6))
    latency_time = (n - 1) * (max_rtt / 1000.0)
    return data_time + latency_time


def simulate_tensor_parallel(
    devices: List[Device],
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    parallel_degree: Optional[int] = None,
    efficiency_loss: float = 0.85,
) -> Dict:
    """
    Tensor-parallel over a group of devices. We divide compute across k devices and add
    an all-reduce per layer per token (very rough) for attention+FFN outputs.
    """
    if parallel_degree is None:
        k = len(devices)
    else:
        k = min(parallel_degree, len(devices))
        devices = devices[:k]

    # Effective compute when splitting across k devices
    per_layer_flops = model.layer_compute_flops_per_token(seq_len)
    per_token_flops = model.num_layers * per_layer_flops

    # Split compute somewhat evenly; take slowest device as the bottleneck per-synchronization
    device_times = []
    for d in devices:
        t = d.compute_time_s(per_token_flops / k) / efficiency_loss
        device_times.append(t)
    compute_time = max(device_times)  # sync happens each layer -> bottleneck per token

    # All-reduce cost per layer (assume 2 collectives per layer)
    red_bytes = model.allreduce_bytes_per_token
    per_layer_red = ring_allreduce_time_s(red_bytes, devices, net) * 2
    comm_time = per_layer_red * model.num_layers

    per_token = compute_time + comm_time
    total = per_token * gen_tokens
    return {
        "policy": "tensor",
        "k": k,
        "per_token_s": per_token,
        "component_times_s": {"compute": compute_time, "allreduce": comm_time},
        "total_latency_s": total,
        "notes": (
            "Ring all-reduce with conservative link estimates; "
            "includes crude efficiency loss."
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
    devs = [
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

    res = run_experiments(devs, MODEL, NET, SEQ_LEN, GEN_TOKENS)
    from pprint import pprint

    pprint(res)
