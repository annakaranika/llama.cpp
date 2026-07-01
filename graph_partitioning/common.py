"""Common definitions for graph partitioning and distributed inference simulation."""

from enum import Enum
from dataclasses import dataclass
import csv
import math
import statistics
from typing import Dict, List, Optional


class CommunicationModel(Enum):
    """Communication model for distributed inference."""

    CLIENT_SERVER = "client_server"  # Client coordinates all communication
    PEER_TO_PEER = "peer_to_peer"  # Servers communicate directly


class PeerToPeerPolicy(Enum):
    """Peer-to-peer communication pattern."""

    CENTRALIZED = "centralized"  # Leader server coordinates all-reduce
    ALL_TO_ALL = "all_to_all"  # All-to-all all-reduce
    SCATTER_GATHER = "scatter_gather"  # Reduce-scatter + all-gather
    RING = "ring"  # Ring all-reduce
    HIERARCHICAL = "hierarchical"  # Hierarchical all-reduce


class AttentionSplitStrategy(Enum):
    """How to split attention heads across devices."""

    KV_HEADS = "kv_heads"


class FFNSplitStrategy(Enum):
    """How to split FFN layers across devices."""

    KV_HEADS = "kv_heads"
    HIDDEN_DIM = "hidden_dim"
    Q_HEADS = "q_heads"


class PartitioningPolicy(Enum):
    """Graph partitioning policy for model layers."""

    SINGLE_NODE = "single_node"  # All on one device
    PIPELINE = "pipeline"  # Layer-wise pipeline parallelism
    TENSOR = "tensor"  # Tensor parallelism with ring all-reduce
    METIS = "metis"  # METIS k-way partitioning of layer graph
    ALPA = "alpa"  # Alpa-style 2D partitioning (pipeline + tensor)
    HYBRID_PP_TP = "hybrid_pp_tp"  # Memory-aware hybrid pipeline+tensor heuristic
    MSCT = "msct"  # Baechi memory-constrained SCT (favorite-child + memory caps)


class MetisDevConsideration(Enum):
    """Device consideration strategy for METIS partitioning."""

    COMPUTE_ONLY = "compute_only"  # Consider only compute capabilities
    MEMORY_ONLY = "memory_only"  # Consider only memory capacities
    NETWORK_ONLY = "network_only"  # Consider only network bandwidths
    CONSTRAINING_RESOURCE = (
        "constraining_resource"  # Consider the constraining resource
    )
    DOMINANT_RESOURCE = "dominant_resource"  # Consider the dominant resource


class DevNetAggregationStrategy(Enum):
    """Network aggregation strategy for a device's communication bandwidth estimation."""

    SUM = "sum"  # Sum of all link bandwidths
    MAX = "max"  # Maximum link bandwidth
    MIN = "min"  # Minimum link bandwidth


class Quantization(Enum):
    """Weight / KV-cache element size, in *bytes per parameter*.

    Values are the effective bytes-per-weight of the corresponding GGUF
    quantization as produced by llama.cpp (bits-per-weight / 8, averaged over
    all tensors for a typical 1B–7B model). They drive the memory-footprint
    (and therefore swap-penalty) model — NOT compute, which the simulator
    accounts for separately via effective GFLOPS.

    The same enum doubles as a KV-cache dtype: FP16/FP32 are exact, and the
    quantized members map to llama.cpp's --cache-type-{k,v} options.
    """

    FP32 = 4.0
    FP16 = 2.0
    Q8_0 = 1.0625    # 8.5 bpw
    Q6_K = 0.82      # 6.56 bpw
    Q5_K_M = 0.71    # 5.68 bpw  (TinyLlama-1.1B Q5_K_M ≈ 745 MiB)
    Q4_K_M = 0.61    # 4.85 bpw
    Q4_0 = 0.57      # 4.55 bpw

    @property
    def bytes_per_param(self) -> float:
        return float(self.value)


# Effective *sequential read* bandwidth (Mbps) of the storage classes found on
# end-user edge devices, for use as ``Device.swap_bandwidth_mbps``. These bound
# how fast weights that overflow DRAM are paged back in (so they only matter
# when a model does not fit in RAM). Values are sequential-read ceilings
# (Mbps = MB/s × 8); real mmap paging is slower because of page-fault/random
# overhead, so treat these as optimistic. Sources: Raspberry Pi microSD
# benchmarks (DDR50 ~40 MB/s read ceiling), JEDEC eMMC/UFS specs, vendor SSD
# datasheets. NB: SD "speed classes" (Class 10 = 10 MB/s *write*) are write
# specs and are NOT read figures.
# TODO(calibration): replace per-device entries with measured mmap paging rates.
STORAGE_PROFILES: Dict[str, float] = {
    "microsd_pi":   320.0,   # ~40 MB/s   — Raspberry Pi / SBC onboard microSD
    "usb3_flash":  1600.0,   # ~200 MB/s  — USB 3.0 SSD / flash boot drive
    "emmc_51":     3200.0,   # ~400 MB/s  — eMMC 5.1: smart hubs, CM4, set-top boxes
    "sata_ssd":    4400.0,   # ~550 MB/s  — SATA III SSD: mini-PCs, laptops
    "ufs_21":      6400.0,   # ~800 MB/s  — UFS 2.1: mid-range phones/tablets
    "ufs_31":     16800.0,   # ~2100 MB/s — UFS 3.1: flagship phones/tablets
    "nvme_gen3":  28000.0,   # ~3500 MB/s — NVMe Gen3: edge PCs / mini-servers
    "nvme_gen4":  56000.0,   # ~7000 MB/s — NVMe Gen4
}


class CostModel(Enum):
    """Which per-op compute-time model :meth:`Device.compute_time_s` uses.

    GFLOPS   — the analytical "gflops way": ``time = flops / (1e9 * gflops)``.
               A single effective-compute number per device. This is the
               historical default and is kept fully available.
    ROOFLINE — measurement-driven: ``time = max(flops / R_compute, bytes / R_bw)``
               where ``R_compute`` is still ``gflops`` (the compute ceiling) and
               ``R_bw`` is the measured memory bandwidth (``mem_bandwidth_gbps``,
               from test-backend-ops, e.g. ~3.5 GB/s on a Pi 4B). The bandwidth
               term makes bandwidth-bound decode (weights streamed once/token)
               realistic without hand-tuning ``gflops`` down, while compute-bound
               prefill stays governed by the FLOPs term — the regime is selected
               automatically by ``max(...)``.
    """

    GFLOPS = "gflops"
    ROOFLINE = "roofline"


@dataclass
class Device:
    """
    Represents a compute device in the distributed inference simulation.
    Attributes:
        name (str): Device identifier.
        gflops (float): Peak FP32 GFLOPs (or "effective" compute units). Under
            CostModel.ROOFLINE this is the compute ceiling ``R_compute``.
        memory_gb (float): Available memory in GB.

    """

    name: str
    gflops: float  # effective/achievable GFLOPs/s (or "effective" compute units)
    memory_gb: float
    # Bandwidth (Mbps) at which weights overflowing DRAM are paged from storage.
    # Defaults to a Raspberry Pi microSD; override per device from the
    # STORAGE_PROFILES table above, e.g.
    #   Device(..., swap_bandwidth_mbps=STORAGE_PROFILES["ufs_31"]).
    # See that table for per-class values, sources, and the measurement TODO.
    swap_bandwidth_mbps: float = STORAGE_PROFILES["microsd_pi"]
    # Cost model and its measured constant. ``mem_bandwidth_gbps`` is the device's
    # effective DRAM read bandwidth R_bw, in **decimal GB/s (1e9 bytes)** — the same
    # unit as ``memory_gb``, the model's memory (model.py), and swap bandwidth, so
    # the measured value drops straight in with no conversion. Only used under
    # ROOFLINE; when None, compute_time_s falls back to GFLOPS.
    cost_model: CostModel = CostModel.GFLOPS
    mem_bandwidth_gbps: Optional[float] = None
    # Optional measured GFLOPS(n) curve: {batch_size n -> effective GFLOPS}. The
    # compute ceiling R_compute is batch-size dependent — it rises from the GEMV
    # (decode, n=1) rate to a peak around n~8, then regresses at large batch as the
    # working set overflows cache (the n=512 point). When set, ``compute_time_s``
    # interpolates R_compute by the forward's ``seq_len`` instead of using the single
    # ``gflops`` scalar, which fixes prefill time decaying with prompt length.
    # Populated from device_profiles.csv's {decode,peak,prefill512} by
    # ``devices_from_profiles``; ``gflops`` stays the scalar fallback.
    gflops_curve: Optional[Dict[int, float]] = None

    def flops_per_sec(self) -> float:
        """Effective FLOPs/s."""
        return 1e9 * self.gflops

    def r_compute_gflops(self, seq_len: Optional[int] = None) -> float:
        """Effective compute ceiling R_compute (GFLOPS) for a forward of batch
        ``seq_len``. Without a measured ``gflops_curve`` (or seq_len), returns the
        scalar ``gflops``. With a curve, log-interpolates between the bracketing
        measured points and clamps outside the measured range — so e.g. a 128-token
        prefill gets a rate between the n~8 peak and the n=512 regressed value rather
        than the optimistic peak. NB: the curve is a pure-matmul (test-backend-ops)
        rate, so it captures the *shape* of the seq-dependence; a residual constant
        offset vs end-to-end prefill is a separate global derate, not modelled here.
        """
        if not self.gflops_curve or seq_len is None:
            return self.gflops
        pts = sorted(self.gflops_curve.items())  # [(n, gflops), ...] ascending n
        if seq_len <= pts[0][0]:
            return pts[0][1]
        if seq_len >= pts[-1][0]:
            return pts[-1][1]
        for (n0, g0), (n1, g1) in zip(pts, pts[1:]):
            if n0 <= seq_len <= n1:
                # linear in log(n): n spans orders of magnitude (1 -> 512)
                t = (math.log(seq_len) - math.log(n0)) / (math.log(n1) - math.log(n0))
                return g0 + t * (g1 - g0)
        return self.gflops  # unreachable (clamped above), keeps type-checkers happy

    def compute_time_s(
        self,
        flops: float,
        memory: float,
        bytes_streamed_gb: Optional[float] = None,
        seq_len: Optional[int] = None,
    ) -> float:
        """Per-forward-pass time given FLOPs and the working-set memory in GB.

        Under ``CostModel.GFLOPS`` (default) the on-device time is purely
        ``flops / (1e9 * gflops)`` — the historical "gflops way".

        Under ``CostModel.ROOFLINE`` it is ``max(compute, bandwidth)`` where the
        bandwidth term is ``bytes_streamed / mem_bandwidth_gbps``. ``bytes_streamed``
        defaults to the DRAM-resident working set (``min(memory, memory_gb)``,
        in GB) — at decode that is the weights+KV read once per token, so the
        term reproduces the measured ``model_bytes / R_bw`` decode wall. Pass
        ``bytes_streamed_gb`` to override (e.g. exact weight bytes). Compute-bound
        prefill is unaffected because its FLOPs term dominates the ``max``.

        When the working set exceeds device DRAM the overflow is paged in from
        secondary storage at ``swap_bandwidth_mbps``. We model this as a
        serial cost added to compute (llama.cpp's mmap fallback on
        single-thread CPU edge devices barely overlaps I/O with compute).

        ``seq_len`` (the forward's batch/prompt length) selects the compute ceiling
        R_compute from a measured ``gflops_curve`` when one is present — peak rate
        for short forwards, the regressed large-batch rate for long prefills — so
        prefill time no longer decays with prompt length. With no curve it is
        ignored and the scalar ``gflops`` is used.
        """
        r_compute = self.r_compute_gflops(seq_len)
        compute_time = flops / (1e9 * r_compute)
        if self.cost_model is CostModel.ROOFLINE and self.mem_bandwidth_gbps:
            streamed = memory if bytes_streamed_gb is None else bytes_streamed_gb
            resident = min(streamed, self.memory_gb)  # DRAM part; overflow paged below
            bandwidth_time = resident / self.mem_bandwidth_gbps
            compute_time = max(compute_time, bandwidth_time)
        if memory <= self.memory_gb:
            return compute_time
        overflow_gb = memory - self.memory_gb
        swap_bw_gbs = self.swap_bandwidth_mbps / 8000.0  # Mbps -> decimal GB/s
        if swap_bw_gbs <= 0:
            return float("inf")
        return compute_time + overflow_gb / swap_bw_gbs

    def repr(self, bw: float) -> str:
        """String representation including, compute, memory and bandwidth."""
        return f"{self.name} ({self.gflops}GFLOPs, {self.memory_gb}GB, {bw}Mbps)"


def devices_from_profiles(
    csv_path: str,
    quant: str = "q4_K",
    compute: str = "peak",
    cost_model: CostModel = CostModel.ROOFLINE,
    default_mem_gb: float = 4.0,
    swap_bandwidth_mbps: float = STORAGE_PROFILES["microsd_pi"],
    uniform_compute: bool = False,
    interp_compute: bool = True,
) -> List[Device]:
    """Build a heterogeneous device pool from a measured ``device_profiles.csv``.

    The CSV is produced by ``rpi-automation/parse_gflops.py`` and has one row per
    device with ``R_bw_GBps`` plus ``<quant>_{decode,peak,prefill512}_gflops``.

    Args:
        csv_path: path to ``device_profiles.csv``.
        quant: which dtype's GFLOPS columns to read (must match the CSV header).
        compute: which GFLOPS column drives the scalar ``gflops`` fallback —
            "peak" (compute ceiling, the right choice for a roofline), "decode", or
            "prefill512" (conservative, bakes in the n=512 cache/throttle regression).
        cost_model: assigned to every device; ROOFLINE uses the measured R_bw.
        default_mem_gb: used when a row's ``mem_gb`` is blank (e.g. legacy runs).
        interp_compute: when True (default), also load the {decode,peak,prefill512}
            GFLOPS as a ``gflops_curve`` anchored at n={1,8,512} so ``compute_time_s``
            interpolates R_compute by the forward's seq_len. The peak anchor n=8 is
            the end of the measured plateau (decline starts after); the 8->512 gap is
            log-interpolated and unmeasured — densify the sweep there to refine it.

    Returns one ``Device`` per row, with ``gflops`` = chosen scalar compute rate,
    ``gflops_curve`` = the 3-point GFLOPS(n) curve (when ``interp_compute``), and
    ``mem_bandwidth_gbps`` = measured ``R_bw_GBps``.
    """
    gflops_col = f"{quant}_{compute}_gflops"
    # n-anchors for the GFLOPS(n) interpolation curve. decode=GEMV (n=1); peak is
    # the plateau measured ~n=2..8 so we anchor it at the high end (8) and let the
    # cache/throttle decline run 8 -> 512.
    curve_cols = {1: f"{quant}_decode_gflops", 8: f"{quant}_peak_gflops",
                  512: f"{quant}_prefill512_gflops"}
    devices: List[Device] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if gflops_col not in (reader.fieldnames or []):
            raise KeyError(
                f"{gflops_col!r} not in {csv_path}; available: {reader.fieldnames}"
            )
        for row in reader:
            name = row.get("device") or row.get("name")
            try:
                gflops = float(row[gflops_col])
            except (TypeError, ValueError):
                print(f"  skipping {name}: no {gflops_col} value")
                continue
            try:
                mem_gb = float(row.get("mem_gb") or default_mem_gb)
            except (TypeError, ValueError):
                mem_gb = default_mem_gb
            try:
                # R_bw is decimal GB/s, the same unit as memory (see model.py) and
                # swap bandwidth — drops straight in, no conversion.
                r_bw = float(row.get("R_bw_GBps") or "")
            except (TypeError, ValueError):
                r_bw = None
            curve = None
            if interp_compute:
                curve = {}
                for n, col in curve_cols.items():
                    try:
                        curve[n] = float(row[col])
                    except (KeyError, TypeError, ValueError):
                        pass
                curve = curve or None  # drop empty curve -> scalar gflops fallback
            devices.append(
                Device(
                    name=name,
                    gflops=gflops,
                    memory_gb=mem_gb,
                    swap_bandwidth_mbps=swap_bandwidth_mbps,
                    cost_model=cost_model,
                    mem_bandwidth_gbps=r_bw,
                    gflops_curve=curve,
                )
            )
    # For a near-identical fleet, the per-device peak-GFLOPS spread is mostly
    # measurement/thermal noise (e.g. one board's peak measured while throttling).
    # Since prefill is compute-bound, that noise makes pipeline prefill non-monotonic
    # in device count. uniform_compute replaces every R_compute with the fleet median
    # so the cluster is treated as compute-homogeneous (R_bw is left per-device).
    if uniform_compute and devices:
        median_gflops = statistics.median(d.gflops for d in devices)
        # also flatten the GFLOPS(n) curve to the per-anchor fleet median, else
        # interpolation would re-introduce the per-device spread we just removed.
        anchors = sorted({n for d in devices if d.gflops_curve for n in d.gflops_curve})
        median_curve = {
            n: statistics.median(
                d.gflops_curve[n] for d in devices if d.gflops_curve and n in d.gflops_curve
            )
            for n in anchors
        } or None
        for d in devices:
            d.gflops = median_gflops
            if d.gflops_curve:
                d.gflops_curve = dict(median_curve) if median_curve else None
    return devices


def total_avail_memory(devices: List[Device]) -> float:
    """
    Compute total memory across devices.
    """
    return sum(dev.memory_gb for dev in devices)


def std_times(times: List[float]) -> float:
    """
    Compute standard deviation of compute times.
    """
    return statistics.stdev(times) if len(times) > 1 else 0.0


def efficiency(times: List[float]) -> float:
    """
    Compute load balance efficiency as mean / max compute time.
    """
    if not times:
        return 0.0
    mean = statistics.mean(times)
    max_time = max(times)
    if max_time == 0:
        return 0.0
    return mean / max_time
