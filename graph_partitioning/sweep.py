"""Parameter sweeps across model and device specs.

Runs every partitioning policy across one or more swept axes and produces:
  - 1D line plots (latency vs parameter, one line per policy)
  - 2D heatmaps (per-policy latency over two axes)

Axes supported out of the box:
  * num_devices  (cluster size)
  * model_size   (scales num_layers / d_model / d_ff / d_k / d_ff and heads)
  * bandwidth    (default link bandwidth in Mbps)
  * seq_len      (prefill length)
  * gen_tokens   (decode length)
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import graph_plots
from .common import Device, PartitioningPolicy
from .compare_policies import _dev_color
from .logging_utils import setup_logger, setup_output_dir
from .model import ModelSpec
from .network import Network
from .policies.alpa import simulate_alpa
from .policies.metis import simulate_metis
from .policies.pipeline import simulate_pipeline
from .policies.hybrid_pp_tp import simulate_hybrid_pp_tp
from .policies.msct import simulate_msct
from .policies.single_node import simulate_single_node
from .policies.tensor import simulate_tensor_parallel

LOGGER = setup_logger(name=__name__)

POLICY_RUNNERS: Dict[str, Callable] = {
    "single_node": simulate_single_node,
    "pipeline": simulate_pipeline,
    "tensor": simulate_tensor_parallel,
    "metis": simulate_metis,
    "alpa": simulate_alpa,
    "hybrid_pp_tp": simulate_hybrid_pp_tp,
    "msct": simulate_msct,
}

POLICY_COLOR: Dict[str, str] = {
    name: _dev_color(i) for i, name in enumerate(POLICY_RUNNERS.keys())
}

POLICY_MARKER: Dict[str, str] = {
    "single_node": "o",
    "pipeline": "s",
    "tensor": "D",
    "metis": "^",
    "alpa": "v",
    "hybrid_pp_tp": "*",
    "msct": "P",
}


# ---------------------------------------------------------------------------
# Quiet runner
# ---------------------------------------------------------------------------


@contextmanager
def quiet_policies():
    """Silence per-policy plotting and redirect their output dirs to a temp dir.

    Without this, every simulate_* call writes a fresh timestamped folder under
    ``sim_output/`` and emits several PDFs — quickly filling the disk during a
    sweep of hundreds of configurations.
    """
    from .policies import hybrid_pp_tp as _hybrid_pp_tp
    from .policies import metis as _metis
    from .policies import msct as _msct
    from .policies import pipeline as _pipeline
    from .policies import tensor as _tensor

    def _noop(*args, **kwargs):  # noqa: ARG001
        return None

    tmpdir = tempfile.mkdtemp(prefix="gp_sweep_")
    counter = {"n": 0}

    def _tmp_setup(_parent: str) -> str:
        counter["n"] += 1
        sub = os.path.join(tmpdir, str(counter["n"]))
        os.makedirs(sub, exist_ok=True)
        return sub

    patched: List[Tuple[object, str, object]] = []

    def _patch(target: object, attr: str, value: object) -> None:
        if hasattr(target, attr):
            patched.append((target, attr, getattr(target, attr)))
            setattr(target, attr, value)

    for mod in (_pipeline, _tensor, _metis, _hybrid_pp_tp, _msct):
        _patch(mod, "draw_partitions", _noop)
        _patch(mod, "draw_device_computation_breakdown", _noop)
        _patch(mod, "plot_resource_utilization", _noop)
        _patch(mod, "setup_output_dir", _tmp_setup)
    # metis's MetisCtx.setup_output_dir calls the module-level setup_output_dir
    # we just patched, so CTX.output_dir is updated correctly without a direct
    # patch here. Patching the method on CTX directly would shadow the class
    # method and leave self.output_dir unset, breaking later writes.

    plt.close("all")
    try:
        yield tmpdir
    finally:
        for target, attr, value in patched:
            setattr(target, attr, value)
        shutil.rmtree(tmpdir, ignore_errors=True)
        plt.close("all")


def run_all_policies_quiet(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict[str, float]:
    """Run every policy on one config and return ``{policy: total_latency_s}``.

    ``kv_cache_seq_len`` is forwarded to policies that accept it so memory
    pressure (and the swap penalty) reflects the actual context length even
    when ``seq_len`` is small (e.g. a single decode step).
    """
    import inspect

    results: Dict[str, float] = {}
    with quiet_policies():
        for name, runner in POLICY_RUNNERS.items():
            try:
                extra = {}
                if (
                    kv_cache_seq_len is not None
                    and "kv_cache_seq_len" in inspect.signature(runner).parameters
                ):
                    extra["kv_cache_seq_len"] = kv_cache_seq_len
                r = runner(model, net, seq_len, gen_tokens, **extra)
                results[name] = float(r.get("total_latency_s", float("nan")))
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Policy %s failed: %s", name, exc)
                results[name] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def default_servers(n: int) -> List[Device]:
    """Generate ``n`` heterogeneous synthetic RPi-class devices."""
    gflops_pool = [1.0, 3.0, 5.5, 0.4, 2.0, 4.0, 6.0, 0.8, 1.5, 7.0]
    memory_pool = [2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 4.0, 2.0, 1.0]
    return [
        Device(
            name=f"rpi-{i + 1}",
            gflops=gflops_pool[i % len(gflops_pool)],
            memory_gb=memory_pool[i % len(memory_pool)],
        )
        for i in range(n)
    ]


def default_network(
    servers: List[Device], bw_mbps: float = 200.0, rtt_ms: float = 18.0
) -> Network:
    """Build a simple symmetric-link Network from a server list."""
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=bw_mbps, default_rtt_ms=rtt_ms)
    return net


def scaled_model(scale: float) -> ModelSpec:
    """Scale TinyLlama dimensions by ``scale`` (rounded), keeping GQA ratios."""
    base = ModelSpec()
    s = max(scale, 0.05)
    num_layers = max(2, int(round(base.num_layers * s)))
    d_model = max(64, int(round(base.d_model * s)))
    # round d_model up to a multiple of (num_heads / num_kv_heads * d_k)?
    # keep d_k constant, scale num_heads and num_kv_heads proportionally so d_model = num_heads * d_k stays consistent
    d_k = base.d_k
    num_heads = max(2, d_model // d_k)
    # keep GQA ratio
    gqa = base.num_heads // base.num_kv_heads
    num_kv_heads = max(1, num_heads // gqa)
    d_ff = max(64, int(round(base.d_ff * s)))
    d_model_aligned = num_heads * d_k
    return ModelSpec(
        name=f"TinyLlama-scaled-x{scale:.2f}",
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_model=d_model_aligned,
        d_k=d_k,
        d_ff=d_ff,
        vocab_size=base.vocab_size,
    )


# ---------------------------------------------------------------------------
# 1D sweeps
# ---------------------------------------------------------------------------


def sweep_axis(
    axis_name: str,
    values: Sequence,
    config_fn: Callable,
) -> Dict[str, List[float]]:
    """Run one axis sweep. ``config_fn(value)`` must return
    ``(model, net, seq_len, gen_tokens)``.

    Returns ``{policy: [latency for each value]}``.
    """
    results: Dict[str, List[float]] = {p: [] for p in POLICY_RUNNERS}
    for v in values:
        model, net, seq_len, gen_tokens = config_fn(v)
        LOGGER.info(
            "%s sweep: value=%s, %d layers, %d devices",
            axis_name,
            v,
            model.num_layers,
            len(net.servers),
        )
        latencies = run_all_policies_quiet(model, net, seq_len, gen_tokens)
        for p in POLICY_RUNNERS:
            results[p].append(latencies.get(p, float("nan")))
    return results


def plot_axis_lines(
    axis_name: str,
    axis_label: str,
    values: Sequence,
    results: Dict[str, List[float]],
    save_path: str,
    log_y: bool = False,
    y_label: str = "total latency (s)",
    metric_name: str = "latency",
    xtick_labels: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Line plot: x = swept value, y = ``y_label``, one line per policy.

    ``xtick_labels`` labels a categorical x-axis (e.g. named model presets);
    pass numeric positions as ``values`` and the names here.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for policy, ys in results.items():
        ax.plot(
            values,
            ys,
            label=policy,
            color=POLICY_COLOR.get(policy, "#444"),
            marker=POLICY_MARKER.get(policy, "o"),
            linewidth=2,
            markersize=7,
        )
    if xtick_labels is not None:
        ax.set_xticks(list(values))
        ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Per-policy {metric_name} vs {axis_label}",
        fontsize=13,
        fontweight="bold",
    )
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved %s sweep to %s", axis_name, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-device memory extraction & plotting
# ---------------------------------------------------------------------------


def extract_device_memory(
    result: Dict, devices: List[Device], model_num_layers: int
) -> Dict[str, float]:
    """Return ``{device_name: memory_gb_used}`` from a policy result dict.

    Unmentioned devices map to 0. Used for the per-device memory plots in
    sweep_model_size.py to show where each policy actually places weight +
    KV cache footprint.
    """
    out = {d.name: 0.0 for d in devices}
    policy = result.get("policy", "")

    if policy == "single_node":
        name = result.get("device", "")
        if name in out:
            try:
                out[name] = float(result.get("mem_gb", 0.0))
            except (TypeError, ValueError):
                out[name] = 0.0

    elif policy in ("pipeline", "hybrid_pp_tp"):
        for stage in result.get("stages", []):
            name = stage.get("device", "")
            if name in out:
                out[name] += float(stage.get("mem_gb", 0.0))

    elif policy == "msct":
        for name, mem in result.get("device_memory_gb", {}).items():
            if name in out:
                out[name] = float(mem)

    elif policy == "tensor":
        # device_allocation stores per-layer memory; multiply by num_layers.
        for name, alloc in result.get("device_allocation", {}).items():
            if name in out:
                out[name] = float(alloc.get("memory_usage_gb", 0.0)) * model_num_layers

    elif policy == "metis":
        # device_workloads.memory_usage_gb is already the total per device.
        for name, w in result.get("device_workloads", {}).items():
            if name in out:
                out[name] = float(w.get("memory_usage_gb", 0.0))

    elif policy == "alpa":
        # Alpa doesn't expose per-device memory; report 0 (transparent gap).
        pass

    return out


def run_all_policies_with_memory(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    kv_cache_seq_len: Optional[int] = None,
) -> Dict[str, Dict]:
    """Run every policy and return ``{policy: {"latency_s": ..., "mem_gb": {dev_name: gb}}}``."""
    import inspect

    out: Dict[str, Dict] = {}
    with quiet_policies():
        for name, runner in POLICY_RUNNERS.items():
            try:
                extra = {}
                if (
                    kv_cache_seq_len is not None
                    and "kv_cache_seq_len" in inspect.signature(runner).parameters
                ):
                    extra["kv_cache_seq_len"] = kv_cache_seq_len
                r = runner(model, net, seq_len, gen_tokens, **extra)
                out[name] = {
                    "latency_s": float(r.get("total_latency_s", float("nan"))),
                    "mem_gb": extract_device_memory(r, net.servers, model.num_layers),
                }
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Policy %s failed: %s", name, exc)
                out[name] = {
                    "latency_s": float("nan"),
                    "mem_gb": {d.name: 0.0 for d in net.servers},
                }
    return out


def plot_device_memory_grid(
    per_scale: List[Dict[str, Dict]],
    scales: List[float],
    devices: List[Device],
    save_path: str,
    x_label: str = "model scale",
) -> plt.Figure:
    """Grid of subplots — one per policy — showing each device's memory
    footprint as the swept axis varies. Dashed horizontal lines mark each
    device's DRAM capacity so swap regimes are visible at a glance.
    """
    if not per_scale:
        raise ValueError("per_scale is empty")
    policy_names = list(per_scale[0].keys())
    n = len(policy_names)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    dev_color = ["#264b7b", "#a6c4e0", "#9467bd", "#e38f45", "#71b171", "#be6c6c"]

    for ax_idx, policy in enumerate(policy_names):
        ax = axes[ax_idx]
        for d_idx, dev in enumerate(devices):
            ys = [scale_entry[policy]["mem_gb"].get(dev.name, 0.0) for scale_entry in per_scale]
            ax.plot(
                scales,
                ys,
                label=dev.name,
                color=dev_color[d_idx % len(dev_color)],
                marker="o",
                markersize=5,
                linewidth=2,
            )
            # Capacity dashed line
            ax.axhline(
                dev.memory_gb,
                linestyle="--",
                linewidth=1,
                color=dev_color[d_idx % len(dev_color)],
                alpha=0.5,
            )
        ax.set_title(policy, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("memory used (GB)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    for ax in axes[len(policy_names) :]:
        ax.set_visible(False)

    # One shared legend for devices + capacity convention
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles + [plt.Line2D([0], [0], color="gray", linestyle="--")],
        labels + ["device DRAM capacity (dashed)"],
        loc="lower center",
        ncol=min(len(devices) + 1, 6),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        "Per-device memory footprint by policy",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved per-device memory plot to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# 2D heatmap
# ---------------------------------------------------------------------------


def sweep_2d(
    x_values: Sequence,
    y_values: Sequence,
    config_fn: Callable,
) -> Dict[str, np.ndarray]:
    """Run a 2D sweep. ``config_fn(x, y)`` -> ``(model, net, seq_len, gen_tokens)``.

    Returns ``{policy: matrix[ny, nx]}``.
    """
    nx_ = len(x_values)
    ny_ = len(y_values)
    out: Dict[str, np.ndarray] = {
        p: np.full((ny_, nx_), np.nan) for p in POLICY_RUNNERS
    }
    for j, yv in enumerate(y_values):
        for i, xv in enumerate(x_values):
            model, net, seq_len, gen_tokens = config_fn(xv, yv)
            LOGGER.info("2D sweep cell (%d,%d) x=%s y=%s", i, j, xv, yv)
            latencies = run_all_policies_quiet(model, net, seq_len, gen_tokens)
            for p in POLICY_RUNNERS:
                out[p][j, i] = latencies.get(p, float("nan"))
    return out


def plot_2d_heatmaps(
    matrices: Dict[str, np.ndarray],
    x_values: Sequence,
    y_values: Sequence,
    x_label: str,
    y_label: str,
    save_path: str,
    title_suffix: str = "",
) -> plt.Figure:
    """One subplot per policy; share axes; per-subplot colour scale."""
    n = len(matrices)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (policy, M) in zip(axes, matrices.items()):
        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[-0.5, len(x_values) - 0.5, -0.5, len(y_values) - 0.5],
        )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values], fontsize=8, rotation=30)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values], fontsize=8)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(policy, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="latency (s)")

    for ax in axes[len(matrices) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-policy latency: {y_label} vs {x_label}{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved 2D heatmap to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# CSV dump helper
# ---------------------------------------------------------------------------


def _dump_axis_csv(
    save_path: str, axis_label: str, values: Sequence, results: Dict[str, List[float]]
) -> None:
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([axis_label] + list(results.keys()))
        for i, v in enumerate(values):
            writer.writerow([v] + [results[p][i] for p in results])


def _dump_2d_csv(
    save_path: str,
    matrices: Dict[str, np.ndarray],
    x_values: Sequence,
    y_values: Sequence,
    x_label: str,
    y_label: str,
) -> None:
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", y_label, x_label, "latency_s"])
        for policy, M in matrices.items():
            for j, yv in enumerate(y_values):
                for i, xv in enumerate(x_values):
                    writer.writerow([policy, yv, xv, M[j, i]])


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_all_sweeps(
    output_dir: Optional[str] = None,
    base_model: Optional[ModelSpec] = None,
    base_servers: Optional[List[Device]] = None,
    base_bw_mbps: float = 200.0,
    base_rtt_ms: float = 18.0,
    base_seq_len: int = 1,
    base_gen_tokens: int = 1,
    device_counts: Sequence[int] = (2, 3, 4, 6, 8),
    models: Optional[Sequence[ModelSpec]] = None,
    bandwidths_mbps: Sequence[float] = (10, 50, 100, 200, 500, 1000),
    seq_lens: Sequence[int] = (1, 16, 64, 256, 1024),
    gen_tokens_list: Sequence[int] = (1, 8, 32, 128),
) -> str:
    """Run all four 1D sweeps + two 2D heatmaps; save PDFs and CSVs.

    Returns the output directory.
    """
    if output_dir is None:
        output_dir = setup_output_dir("sim_output/sweep")
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info("Writing sweep artefacts to %s", output_dir)

    if base_model is None:
        base_model = ModelSpec()
    if base_servers is None:
        base_servers = default_servers(4)
    if models is None:
        models = [base_model]

    # --- 1D: number of devices ---
    def cfg_devices(n):
        servers = default_servers(int(n))
        net = default_network(servers, base_bw_mbps, base_rtt_ms)
        return base_model, net, base_seq_len, base_gen_tokens

    r_dev = sweep_axis("num_devices", device_counts, cfg_devices)
    plot_axis_lines(
        "num_devices",
        "number of devices",
        device_counts,
        r_dev,
        save_path=os.path.join(output_dir, "sweep_num_devices.pdf"),
    )
    _dump_axis_csv(
        os.path.join(output_dir, "sweep_num_devices.csv"),
        "num_devices",
        device_counts,
        r_dev,
    )

    # --- 1D: model (named presets: TinyLlama / Llama-7B / Llama-13B) ---
    # Size the cluster so the LARGEST preset fits (else single_node/msct would
    # swap or fail to place it). Grow the heterogeneous pool until total DRAM
    # comfortably covers the biggest model's footprint.
    largest_mem = max(m.total_memory_usage() for m in models)
    n_model_devs = max(len(base_servers), len(device_counts) and max(device_counts))
    while sum(d.memory_gb for d in default_servers(n_model_devs)) < 1.3 * largest_mem:
        n_model_devs += 1
    model_servers = default_servers(n_model_devs)
    model_by_name = {m.name: m for m in models}

    def cfg_model(name):
        net = default_network(model_servers, base_bw_mbps, base_rtt_ms)
        return model_by_name[name], net, base_seq_len, base_gen_tokens

    model_names = [m.name for m in models]
    r_model = sweep_axis("model", model_names, cfg_model)
    plot_axis_lines(
        "model",
        "model",
        list(range(len(models))),
        r_model,
        save_path=os.path.join(output_dir, "sweep_model_size.pdf"),
        log_y=True,
        xtick_labels=model_names,
    )
    _dump_axis_csv(
        os.path.join(output_dir, "sweep_model_size.csv"),
        "model",
        model_names,
        r_model,
    )

    # --- 1D: bandwidth ---
    def cfg_bw(bw):
        net = default_network(base_servers, float(bw), base_rtt_ms)
        return base_model, net, base_seq_len, base_gen_tokens

    r_bw = sweep_axis("bandwidth_mbps", bandwidths_mbps, cfg_bw)
    plot_axis_lines(
        "bandwidth_mbps",
        "link bandwidth (Mbps)",
        bandwidths_mbps,
        r_bw,
        save_path=os.path.join(output_dir, "sweep_bandwidth.pdf"),
        log_y=True,
    )
    _dump_axis_csv(
        os.path.join(output_dir, "sweep_bandwidth.csv"),
        "bandwidth_mbps",
        bandwidths_mbps,
        r_bw,
    )

    # --- 1D: seq_len ---
    def cfg_seq(sl):
        net = default_network(base_servers, base_bw_mbps, base_rtt_ms)
        return base_model, net, int(sl), base_gen_tokens

    r_seq = sweep_axis("seq_len", seq_lens, cfg_seq)
    plot_axis_lines(
        "seq_len",
        "sequence length (prefill tokens)",
        seq_lens,
        r_seq,
        save_path=os.path.join(output_dir, "sweep_seq_len.pdf"),
        log_y=True,
    )
    _dump_axis_csv(
        os.path.join(output_dir, "sweep_seq_len.csv"), "seq_len", seq_lens, r_seq
    )

    # --- 1D: gen_tokens ---
    def cfg_gen(gt):
        net = default_network(base_servers, base_bw_mbps, base_rtt_ms)
        return base_model, net, base_seq_len, int(gt)

    r_gen = sweep_axis("gen_tokens", gen_tokens_list, cfg_gen)
    plot_axis_lines(
        "gen_tokens",
        "generation tokens (decode)",
        gen_tokens_list,
        r_gen,
        save_path=os.path.join(output_dir, "sweep_gen_tokens.pdf"),
        log_y=True,
    )
    _dump_axis_csv(
        os.path.join(output_dir, "sweep_gen_tokens.csv"),
        "gen_tokens",
        gen_tokens_list,
        r_gen,
    )

    # --- 2D: devices × bandwidth ---
    def cfg_dev_bw(n, bw):
        servers = default_servers(int(n))
        net = default_network(servers, float(bw), base_rtt_ms)
        return base_model, net, base_seq_len, base_gen_tokens

    M_dev_bw = sweep_2d(device_counts, bandwidths_mbps, cfg_dev_bw)
    plot_2d_heatmaps(
        M_dev_bw,
        x_values=device_counts,
        y_values=bandwidths_mbps,
        x_label="num devices",
        y_label="bandwidth (Mbps)",
        save_path=os.path.join(output_dir, "heatmap_devices_x_bandwidth.pdf"),
    )
    _dump_2d_csv(
        os.path.join(output_dir, "heatmap_devices_x_bandwidth.csv"),
        M_dev_bw,
        device_counts,
        bandwidths_mbps,
        "num_devices",
        "bandwidth_mbps",
    )

    # --- 2D: seq_len × gen_tokens ---
    def cfg_seq_gen(sl, gt):
        net = default_network(base_servers, base_bw_mbps, base_rtt_ms)
        return base_model, net, int(sl), int(gt)

    M_seq_gen = sweep_2d(seq_lens, gen_tokens_list, cfg_seq_gen)
    plot_2d_heatmaps(
        M_seq_gen,
        x_values=seq_lens,
        y_values=gen_tokens_list,
        x_label="seq_len",
        y_label="gen_tokens",
        save_path=os.path.join(output_dir, "heatmap_seqlen_x_gentokens.pdf"),
    )
    _dump_2d_csv(
        os.path.join(output_dir, "heatmap_seqlen_x_gentokens.csv"),
        M_seq_gen,
        seq_lens,
        gen_tokens_list,
        "seq_len",
        "gen_tokens",
    )

    LOGGER.info("All sweeps complete; outputs in %s", output_dir)
    return output_dir
