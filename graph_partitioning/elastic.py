"""Elastic (KV-growth-driven) pipeline parallelism for the WiFi Pi cluster.

Within a single request the only memory term that changes is the KV cache, which
grows linearly with position ``t``. So the per-device footprint is

    M_d(t) = W_d (weights, static)  +  slope_d · t (KV)  +  A_d (activations)

and a device OOMs once ``M_d(t) > B``. This module turns that into an *elastic*
schedule: start with the minimal pipeline that holds the weights + a short context,
and **recruit a device** (or shift a boundary) as ``t`` grows, so the cluster's
device footprint grows *with the sequence* instead of being reserved for the max
context up front. TP never appears — KV distributes as total/N in both PP and TP,
so growth is a pipeline-rebalancing problem (see ADAPTIVE_PARALLELISM.md).

The costed knob is **KV migration**: weights are free to "move" (each device caches
all slices locally, so a re-partition is a re-mmap, not a WiFi re-upload), but the
accumulated KV of any reassigned layer must travel the mesh. Two primitives:
  * ``boundary_shift`` — the recruit takes over only its layer share, peeled from a
    neighbour: moves ``total_KV(t)/N_to``.
  * ``reshard``        — re-even-split everything: moves ``total_KV(t)·(N_to-1)/N_to``.
With ``prefetch_tokens`` the recruit is triggered *early* so the migration overlaps
ongoing decode and the stall is (partly) hidden.

Run a demo:  PYTHONPATH=. python3 graph_partitioning/elastic.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from graph_partitioning.model import ModelSpec


# ── memory / schedule primitives (pure, network-free) ────────────────────────

def per_layer_weight_gb(model: ModelSpec) -> float:
    """Weights (GB) of one full transformer layer (all KV heads, full FFN)."""
    return (model.attn_weights_gb_per_kv_head() * model.num_kv_heads
            + model.ffn_memory_gb_per_fraction(1.0))


def kv_slope_gb_per_layer_token(model: ModelSpec) -> float:
    """KV growth (GB per token) of one full layer."""
    return model.kv_cache_gb_per_kv_head(1) * model.num_kv_heads


def heaviest_stage_layers(num_layers: int, n: int) -> int:
    """Layers on the most-loaded stage of an even ``n``-way pipeline split."""
    return math.ceil(num_layers / n)


def stage_weight_gb(model: ModelSpec, n: int, with_embedding: bool = True) -> float:
    """Static weight footprint (GB) of the heaviest stage at ``n`` devices.

    The embedding/output matrix rides one end stage; charged to the heaviest
    stage here (conservative for the t_max bound)."""
    w = per_layer_weight_gb(model) * heaviest_stage_layers(model.num_layers, n)
    if with_embedding:
        w += model.embedding_weights_gb()
    return w


def t_max(model: ModelSpec, n: int, budget_gb: float, activation_gb: float = 0.07) -> int:
    """Max context (tokens) before the heaviest stage exceeds ``budget_gb`` at
    ``n`` devices. Returns -1 if the weights alone don't fit."""
    head = budget_gb - stage_weight_gb(model, n) - activation_gb
    if head <= 0:
        return -1
    slope = kv_slope_gb_per_layer_token(model) * heaviest_stage_layers(model.num_layers, n)
    return int(head / slope)


def min_devices_for_weights(model: ModelSpec, budget_gb: float, pool: int,
                            activation_gb: float = 0.07) -> Optional[int]:
    for n in range(1, pool + 1):
        if stage_weight_gb(model, n) + activation_gb < budget_gb:
            return n
    return None


def min_devices_for_context(model: ModelSpec, context: int, budget_gb: float, pool: int,
                            activation_gb: float = 0.07) -> Optional[int]:
    for n in range(1, pool + 1):
        if t_max(model, n, budget_gb, activation_gb) >= context:
            return n
    return None


@dataclass
class Segment:
    t_start: int
    t_end: int
    n: int


@dataclass
class Recruit:
    t: int          # context position at which the device joins
    n_from: int
    n_to: int


@dataclass
class Schedule:
    segments: List[Segment]
    recruits: List[Recruit]
    feasible: bool          # did we reach L_max within the pool?
    reached: int            # max context actually reachable


def recruitment_schedule(model: ModelSpec, L_max: int, budget_gb: float, pool: int,
                         *, prefill_len: int = 0, activation_gb: float = 0.07) -> Schedule:
    """Elastic schedule from ``prefill_len`` to ``L_max``: minimal N that fits the
    current context, recruiting one device each time KV growth hits ``t_max(N)``."""
    n = min_devices_for_context(model, prefill_len, budget_gb, pool, activation_gb)
    if n is None:                                        # can't even hold the prompt
        n0 = min_devices_for_weights(model, budget_gb, pool, activation_gb)
        return Schedule([], [], False, t_max(model, n0, budget_gb, activation_gb) if n0 else -1)
    segments: List[Segment] = []
    recruits: List[Recruit] = []
    t = prefill_len
    while True:
        reach = t_max(model, n, budget_gb, activation_gb)
        seg_end = min(reach, L_max)
        segments.append(Segment(t, seg_end, n))
        if seg_end >= L_max:
            return Schedule(segments, recruits, True, L_max)
        if n >= pool:                                    # out of devices
            return Schedule(segments, recruits, False, reach)
        recruits.append(Recruit(reach, n, n + 1))
        n += 1
        t = reach


# ── migration cost ───────────────────────────────────────────────────────────

def total_kv_gb(model: ModelSpec, t: int) -> float:
    """Total KV (GB) across all layers at context ``t`` (undistributed)."""
    return kv_slope_gb_per_layer_token(model) * model.num_layers * t


def migration_gb(model: ModelSpec, n_to: int, t: int, mode: str = "boundary_shift") -> float:
    """KV bytes (GB) that must travel the mesh to grow from ``n_to-1`` to ``n_to``
    devices at context ``t``."""
    kv = total_kv_gb(model, t)
    if mode == "reshard":
        return kv * (n_to - 1) / n_to          # most slices reshuffle
    return kv / n_to                            # recruit's share only, peeled from a neighbour


# ── elastic simulation ───────────────────────────────────────────────────────

@dataclass
class ElasticResult:
    schedule: Schedule
    decode_s: float          # pure per-token decode time, summed
    stall_s: float           # migration stalls not hidden by prefetch
    total_s: float           # decode_s + stall_s
    device_tokens: float     # Σ N(t) dt  — elastic device-time
    # static baseline: constant N sized for L_max
    static_n: Optional[int]
    static_total_s: float
    static_device_tokens: float


def simulate_elastic_decode(
    model: ModelSpec,
    L_max: int,
    *,
    budget_gb: float,
    pool: int,
    decode_latency_fn: Callable[[int, int], float],   # (n_devices, context) -> s/token
    migrate_time_fn: Callable[[float, int, str], float],  # (gb, n_to, mode) -> s
    prefill_len: int = 0,
    activation_gb: float = 0.07,
    mode: str = "boundary_shift",
    prefetch_tokens: int = 0,
) -> ElasticResult:
    """Decode ``prefill_len -> L_max`` under the elastic schedule and compare to a
    static pipeline sized for ``L_max``.

    ``decode_latency_fn(n, ctx)`` is the per-token pipeline decode latency at ``n``
    devices and context ``ctx`` (wire in ``simulate_pipeline``); it's evaluated at
    each segment's midpoint so decode slow-down at long context is captured.
    ``prefetch_tokens`` recruits the device early so migration overlaps decode.
    """
    sched = recruitment_schedule(model, L_max, budget_gb, pool,
                                 prefill_len=prefill_len, activation_gb=activation_gb)

    decode_s = 0.0
    device_tokens = 0.0
    for seg in sched.segments:
        toks = seg.t_end - seg.t_start
        if toks <= 0:
            continue
        mid = (seg.t_start + seg.t_end) // 2
        decode_s += toks * decode_latency_fn(seg.n, mid)
        device_tokens += toks * seg.n

    stall_s = 0.0
    for rc in sched.recruits:
        gb = migration_gb(model, rc.n_to, rc.t, mode)
        migr = migrate_time_fn(gb, rc.n_to, mode)
        # prefetch: run the migration during the last ``prefetch_tokens`` decoded on
        # the old N; only the un-overlapped remainder stalls the stream.
        hidden = prefetch_tokens * decode_latency_fn(rc.n_from, rc.t) if prefetch_tokens else 0.0
        stall_s += max(0.0, migr - hidden)

    total_s = decode_s + stall_s

    # static baseline: one N held for the whole request, sized for L_max.
    static_n = min_devices_for_context(model, L_max, budget_gb, pool, activation_gb)
    if static_n is None:
        static_total = float("inf")
        static_dt = float("inf")
    else:
        s = 0.0
        # same per-token model, constant N, no stalls
        span = L_max - prefill_len
        s = span * decode_latency_fn(static_n, (prefill_len + L_max) // 2)
        static_total = s
        static_dt = span * static_n

    return ElasticResult(sched, decode_s, stall_s, total_s, device_tokens,
                         static_n, static_total, static_dt)


# ── demo ─────────────────────────────────────────────────────────────────────

def _demo():
    from graph_partitioning.common import (
        CommunicationModel, CostModel, Quantization, devices_from_profiles,
    )
    from graph_partitioning.network import Network
    from graph_partitioning.evaluate import DecodeModelSpec
    from graph_partitioning.policies.pipeline import simulate_pipeline

    LLAMA_7B = ModelSpec(name="Llama-7B", num_layers=32, num_heads=32, num_kv_heads=32,
                         d_model=4096, d_k=128, d_ff=11008, vocab_size=32000,
                         weight_quant=Quantization.Q4_K_M, kv_quant=Quantization.FP16)
    LLAMA_13B = ModelSpec(name="Llama-13B", num_layers=40, num_heads=40, num_kv_heads=40,
                          d_model=5120, d_k=128, d_ff=13824, vocab_size=32000,
                          weight_quant=Quantization.Q4_K_M, kv_quant=Quantization.FP16)

    PROFILE = "graph_partitioning/tests/device_profiles.csv"
    RPIS = ["rpi24", "rpi25", "rpi20", "rpi22", "rpi1", "rpi2", "rpi3", "rpi4"]
    pool_map = {d.name: d for d in devices_from_profiles(
        PROFILE, quant="q4_K", compute="peak", cost_model=CostModel.ROOFLINE,
        uniform_compute=True, default_mem_gb=2.0, swap_bandwidth_mbps=320.0)}
    POOL = [pool_map[n] for n in RPIS if n in pool_map]
    BUDGET = 1.64          # GB usable per Pi (measured avail)
    ACT = 0.07             # measured per-device compute buffer (batch1: ~69 MiB); 0.3 was too high
    # Batch-1 HW check (7B -sm layer): N=3 fits (c>=1024), N=4 holds >3072 tok -> the old
    # ACT=0.3 was too conservative (marked N=3 infeasible). ACT=0.07 reproduces both.
    LINKS = "graph_partitioning/tests/pair_links_measured.csv"

    def make_net(n):
        net = Network(servers=POOL[:n], coordinator_name=None,
                      default_bw_mbps=42.0, default_rtt_ms=20.0)
        net.set_pairwise_links(filename=LINKS)
        net.set_comm_config(communication_model=CommunicationModel.CLIENT_SERVER)
        net.rpc_efficiency = 0.7
        return net

    _net_cache: dict = {}
    def decode_latency_fn(n, ctx):
        net = _net_cache.setdefault(n, make_net(n))
        return simulate_pipeline(DecodeModelSpec(MODEL, max(ctx, 1)), net,
                                 seq_len=1, gen_tokens=1)["total_latency_s"]

    def migrate_time_fn(gb, n_to, mode):
        # KV travels the mesh; boundary_shift is ~1 sender (uncontended), reshard is
        # concurrent (airtime-scaled over n_to stations).
        net = _net_cache.setdefault(n_to, make_net(n_to))
        a, b = POOL[0].name, POOL[min(1, n_to - 1)].name
        t = net.xfer_time_s(gb * 1e9, a, b)
        return t * (net._airtime_factor(n_to) if mode == "reshard" else 1.0)

    for MODEL, L_MAX in [(LLAMA_7B, 4096), (LLAMA_13B, 4096)]:
        print(f"\n{'='*72}\n{MODEL.name}  (Q4, budget {BUDGET} GB/dev, act {ACT} GB, pool {len(POOL)})\n{'='*72}")
        print(f"  weights total ≈ {per_layer_weight_gb(MODEL)*MODEL.num_layers + MODEL.embedding_weights_gb():.2f} GB"
              f"   KV ≈ {kv_slope_gb_per_layer_token(MODEL)*MODEL.num_layers*1e3:.2f} MB/token")
        print(f"  {'N':>2} {'t_max(N) tokens':>16} {'stage wt GB':>12}")
        for n in range(1, len(POOL) + 1):
            print(f"  {n:>2} {t_max(MODEL, n, BUDGET, ACT):>16} {stage_weight_gb(MODEL, n):>12.2f}")

        for mode in ("boundary_shift", "reshard"):
            for pf in (0, 256):
                r = simulate_elastic_decode(
                    MODEL, L_MAX, budget_gb=BUDGET, pool=len(POOL),
                    decode_latency_fn=decode_latency_fn, migrate_time_fn=migrate_time_fn,
                    activation_gb=ACT, mode=mode, prefetch_tokens=pf)
                if not r.schedule.segments:
                    print(f"  [{mode}/pf{pf}] INFEASIBLE — prompt doesn't fit; reachable ctx={r.schedule.reached}")
                    continue
                joins = ", ".join(f"N{rc.n_from}->{rc.n_to}@{rc.t}" for rc in r.schedule.recruits) or "none"
                feas = "reached" if r.schedule.feasible else f"STALLED@{r.schedule.reached}"
                print(f"\n  [{mode}, prefetch={pf}]  recruits: {joins}   ({feas} {L_MAX})")
                print(f"     elastic: decode {r.decode_s:6.1f}s + stall {r.stall_s:6.1f}s = {r.total_s:7.1f}s"
                      f"   device·tokens {r.device_tokens:,.0f}")
                if r.static_n:
                    dt_save = 100 * (1 - r.device_tokens / r.static_device_tokens)
                    ovh = 100 * (r.total_s / r.static_total_s - 1)
                    print(f"     static N={r.static_n}: {r.static_total_s:7.1f}s   device·tokens {r.static_device_tokens:,.0f}"
                          f"   ->  elastic saves {dt_save:4.1f}% device·time at {ovh:+.1f}% latency")
                else:
                    print(f"     static: INFEASIBLE for L_max within pool")


if __name__ == "__main__":
    _demo()
