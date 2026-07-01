"""Run every partitioning policy and emit a unified set of comparison plots.

The core artefact is a (num_devices, num_layers) matrix M per policy, where
M[d, l] is the fraction of layer l's compute assigned to device d. The matrix
is plotted as a stacked bar (one bar per layer, segments coloured by device)
so that every policy is visualised in a comparable way.

Usage:
    from graph_partitioning.compare_policies import plot_all_policies
    plot_all_policies(model, net, seq_len=128, gen_tokens=16)
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    AttentionSplitStrategy,
    Device,
    FFNSplitStrategy,
    PartitioningPolicy,
)
from . import graph_plots
from .logging_utils import setup_logger, setup_output_dir
from .model import ModelSpec
from .network import Network
from .policies.alpa import simulate_alpa
from .policies.hybrid_pp_tp import simulate_hybrid_pp_tp
from .policies.metis import simulate_metis
from .policies.msct import simulate_msct
from .policies.pipeline import simulate_pipeline
from .policies.single_node import simulate_single_node
from .policies.tensor import simulate_tensor_parallel

LOGGER = setup_logger(name=__name__)

# 20 distinct colors (tab20) so device colors don't repeat for clusters up to
# 20 devices — a repeat used to make one device's contiguous layers look split.
DEVICE_COLORS = graph_plots.device_palette(20)


def _dev_color(i: int):
    return DEVICE_COLORS[i % len(DEVICE_COLORS)]


def _normalize_columns(M: np.ndarray) -> np.ndarray:
    """Normalize each column to sum to 1 where it has any mass."""
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    return M / col_sums


def layer_device_matrix(
    result: Dict, num_layers: int, devices: List[Device]
) -> np.ndarray:
    """Extract a (num_devices, num_layers) workload-share matrix from a policy result."""
    num_devs = len(devices)
    M = np.zeros((num_devs, num_layers))
    name_to_idx = {d.name: i for i, d in enumerate(devices)}
    policy = result.get("policy", "")

    if policy == "single_node":
        idx = name_to_idx.get(result.get("device", ""))
        if idx is not None:
            M[idx, :] = 1.0

    elif policy == "pipeline":
        for stage in result.get("stages", []):
            idx = name_to_idx.get(stage.get("device"))
            if idx is None:
                continue
            for layer in stage.get("layers", []):
                if 0 <= layer < num_layers:
                    M[idx, layer] = 1.0

    elif policy == "hybrid_pp_tp":
        # Hybrid pipeline+tensor: each stage is a group of devices that
        # tensor-split that stage's layers, so every group device owns a
        # 1/split share of each of the stage's layers (split=1 => pure pipeline).
        for stage in result.get("stages", []):
            group = stage.get("devices", [])
            split = max(len(group), 1)
            for dev_name in group:
                idx = name_to_idx.get(dev_name)
                if idx is None:
                    continue
                for layer in stage.get("layers", []):
                    if 0 <= layer < num_layers:
                        M[idx, layer] = 1.0 / split

    elif policy == "tensor":
        for dev_name, alloc in result.get("device_allocation", {}).items():
            idx = name_to_idx.get(dev_name)
            if idx is None:
                continue
            M[idx, :] = float(alloc.get("ffn_fraction", 0.0))
        M = _normalize_columns(M)

    elif policy == "metis":
        partitions = result.get("partitions")
        if partitions is not None:
            for dev_idx, partition_nodes in enumerate(partitions):
                if dev_idx >= num_devs:
                    continue
                for node in partition_nodes:
                    layer = int(node.get("layer", -1))
                    if 0 <= layer < num_layers:
                        M[dev_idx, layer] += 1.0
        else:
            for dev_name, w in result.get("device_workloads", {}).items():
                idx = name_to_idx.get(dev_name)
                if idx is None:
                    continue
                layers_involved = w.get("layers_involved", set())
                if isinstance(layers_involved, str):
                    layers_involved = {
                        int(x) for x in layers_involved.split(",") if x
                    }
                total_blocks = int(w.get("attention_blocks", 0)) + int(
                    w.get("ffn_blocks", 0)
                )
                if not layers_involved or total_blocks == 0:
                    continue
                share = total_blocks / max(len(layers_involved), 1)
                for layer in layers_involved:
                    if 0 <= layer < num_layers:
                        M[idx, layer] += share
        M = _normalize_columns(M)

    elif policy == "alpa":
        # Intra-op ILP shards/replicates each operator across the selected mesh;
        # every layer involves all cluster devices, so each owns an equal 1/k
        # share (column-normalisation makes replicate vs shard look the same here).
        plan = result.get("selected_plan", {}) or {}
        cluster_devs = plan.get("devices", []) or []
        if cluster_devs:
            share = 1.0 / len(cluster_devs)
            for dev_name in cluster_devs:
                idx = name_to_idx.get(dev_name)
                if idx is not None:
                    M[idx, :] = share

    elif policy == "msct":
        # m-SCT places each operator (attn head / ffn shard) on a device; a
        # device's share of a layer is its fraction of that layer's compute.
        for p in result.get("placement", []):
            idx = name_to_idx.get(p.get("device"))
            layer = int(p.get("layer", -1))
            if idx is None or not (0 <= layer < num_layers):
                continue
            M[idx, layer] += float(p.get("flops", 0.0))
        M = _normalize_columns(M)

    return M


def plot_partition_stacked_bars(
    M: np.ndarray,
    devices: List[Device],
    policy_name: str,
    ax: plt.Axes,
    show_xticks: bool = True,
) -> None:
    """Draw a stacked bar (one bar per layer) on the given axis."""
    num_devs, num_layers = M.shape
    bottom = np.zeros(num_layers)
    layers = np.arange(num_layers)
    for d in range(num_devs):
        ax.bar(
            layers,
            M[d, :],
            bottom=bottom,
            width=0.92,
            color=_dev_color(d),
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += M[d, :]

    ax.set_title(policy_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("workload share")
    ax.set_ylim(0, max(1.05, bottom.max() + 0.05))
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.grid(axis="y", alpha=0.2)
    if show_xticks:
        step = max(1, num_layers // 20)
        ax.set_xticks(layers[::step])
        ax.set_xlabel("layer index")
    else:
        ax.set_xticks([])


def plot_all_policy_partitions(
    results: Dict[str, Dict],
    devices: List[Device],
    num_layers: int,
    save_path: str,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Stacked-bar layer assignment for every policy in a single figure."""
    policy_names = list(results.keys())
    n = len(policy_names)
    if n == 0:
        raise ValueError("No policy results to plot")

    if figsize is None:
        figsize = (max(10.0, num_layers * 0.4), 2.4 * n + 1.5)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, pname) in enumerate(zip(axes, policy_names)):
        M = layer_device_matrix(results[pname], num_layers, devices)
        plot_partition_stacked_bars(
            M, devices, pname, ax, show_xticks=(i == n - 1)
        )

    handles = [
        mpatches.Patch(color=_dev_color(i), label=d.name)
        for i, d in enumerate(devices)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(devices), 6),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        f"Layer-to-device assignment by policy "
        f"({num_layers} layers, {len(devices)} devices)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if graph_plots.SAVE_FIGURES:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved policy partition comparison to %s", save_path)
    return fig


def plot_latency_comparison(
    latencies: Dict[str, float],
    save_path: str,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Bar chart of per-policy total latency."""
    fig, ax = plt.subplots(figsize=figsize)
    policies = list(latencies.keys())
    values = [latencies[p] for p in policies]
    colors = [_dev_color(i) for i in range(len(policies))]
    bars = ax.bar(policies, values, color=colors)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("total latency (s)")
    ax.set_title("Per-policy total latency", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if graph_plots.SAVE_FIGURES:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved per-policy latency comparison to %s", save_path)
    return fig


def plot_device_workload_share(
    results: Dict[str, Dict],
    devices: List[Device],
    num_layers: int,
    save_path: str,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Per-policy stacked bar of how the total workload is divided across devices."""
    fig, ax = plt.subplots(figsize=figsize)
    policy_names = list(results.keys())
    n_devs = len(devices)

    shares = np.zeros((n_devs, len(policy_names)))
    for j, pname in enumerate(policy_names):
        M = layer_device_matrix(results[pname], num_layers, devices)
        shares[:, j] = M.sum(axis=1)
        total = shares[:, j].sum()
        if total > 0:
            shares[:, j] /= total

    bottom = np.zeros(len(policy_names))
    x = np.arange(len(policy_names))
    for d in range(n_devs):
        ax.bar(
            x,
            shares[d, :],
            bottom=bottom,
            color=_dev_color(d),
            edgecolor="white",
            linewidth=0.5,
            label=devices[d].name,
        )
        for j, share in enumerate(shares[d, :]):
            if share > 0.04:
                ax.text(
                    j,
                    bottom[j] + share / 2,
                    f"{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        bottom += shares[d, :]

    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15)
    ax.set_ylabel("share of total workload")
    ax.set_title(
        "Workload distribution per device, by policy",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_devs, 6),
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if graph_plots.SAVE_FIGURES:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    LOGGER.info("Saved device workload share to %s", save_path)
    return fig


def _default_runners(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    attn_split: AttentionSplitStrategy,
    ffn_split: FFNSplitStrategy,
) -> Dict[PartitioningPolicy, Callable[[], Dict]]:
    return {
        PartitioningPolicy.SINGLE_NODE: lambda: simulate_single_node(
            model, net, seq_len, gen_tokens
        ),
        PartitioningPolicy.PIPELINE: lambda: simulate_pipeline(
            model, net, seq_len, gen_tokens
        ),
        PartitioningPolicy.TENSOR: lambda: simulate_tensor_parallel(
            model,
            net,
            seq_len,
            gen_tokens,
            attn_split_strategy=attn_split,
            ffn_split_strategy=ffn_split,
        ),
        PartitioningPolicy.METIS: lambda: simulate_metis(
            model,
            net,
            seq_len,
            gen_tokens,
            attn_split_strategy=attn_split,
            ffn_split_strategy=ffn_split,
        ),
        PartitioningPolicy.ALPA: lambda: simulate_alpa(
            model, net, seq_len, gen_tokens
        ),
        PartitioningPolicy.HYBRID_PP_TP: lambda: simulate_hybrid_pp_tp(
            model,
            net,
            seq_len,
            gen_tokens,
            attn_split_strategy=attn_split,
            ffn_split_strategy=ffn_split,
        ),
        PartitioningPolicy.MSCT: lambda: simulate_msct(
            model,
            net,
            seq_len,
            gen_tokens,
            attn_split_strategy=attn_split,
            ffn_split_strategy=ffn_split,
        ),
    }


def _json_safe(obj):
    from enum import Enum

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return obj.value
    return obj


def plot_all_policies(
    model: ModelSpec,
    net: Network,
    seq_len: int,
    gen_tokens: int,
    output_dir: Optional[str] = None,
    policies: Optional[List[PartitioningPolicy]] = None,
    attn_split: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
    ffn_split: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
) -> Tuple[Dict[str, Dict], str]:
    """Run every configured policy and write a unified set of comparison plots.

    Produces, under ``output_dir``:
      - ``policy_partition_heatmaps.pdf``: stacked bars of per-layer device share
      - ``policy_latency_comparison.pdf``: total latency bar chart
      - ``policy_device_workload_share.pdf``: per-device aggregate share
      - ``policy_results.json``: raw per-policy result dicts

    Returns ``(results, output_dir)``.
    """
    if policies is None:
        policies = [
            PartitioningPolicy.SINGLE_NODE,
            PartitioningPolicy.PIPELINE,
            PartitioningPolicy.TENSOR,
            PartitioningPolicy.METIS,
            PartitioningPolicy.ALPA,
            PartitioningPolicy.HYBRID_PP_TP,
            PartitioningPolicy.MSCT,
        ]

    if output_dir is None:
        output_dir = setup_output_dir("sim_output/compare")
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.info(
        "Running %d policies (seq_len=%d, gen_tokens=%d); outputs -> %s",
        len(policies),
        seq_len,
        gen_tokens,
        output_dir,
    )

    runners = _default_runners(
        model, net, seq_len, gen_tokens, attn_split, ffn_split
    )

    results: Dict[str, Dict] = {}
    for policy in policies:
        if policy not in runners:
            LOGGER.warning("No runner registered for policy %s; skipping", policy)
            continue
        try:
            LOGGER.info("=== Running policy: %s ===", policy.value)
            results[policy.value] = runners[policy]()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Policy %s failed: %s", policy.value, exc)

    if not results:
        raise RuntimeError("No policies produced a result; nothing to plot")

    # Capture each fig and hand it to IPython.display when available so the
    # plots render inline even though this function is called mid-cell (i.e.
    # not as the cell's final expression). Falls back to plt.show() outside
    # Jupyter — the pyplot inline backend's show() is unreliable mid-cell.
    try:
        from IPython.display import display as _display
    except ImportError:
        _display = None

    def _emit(fig):
        if _display is not None:
            _display(fig)
        else:
            plt.show()
        plt.close(fig)

    fig1 = plot_all_policy_partitions(
        results,
        devices=net.servers,
        num_layers=model.num_layers,
        save_path=os.path.join(output_dir, "policy_partition_heatmaps.pdf"),
    )
    _emit(fig1)

    latencies = {
        p: float(r.get("total_latency_s", 0.0)) for p, r in results.items()
    }
    fig2 = plot_latency_comparison(
        latencies,
        save_path=os.path.join(output_dir, "policy_latency_comparison.pdf"),
    )
    _emit(fig2)

    fig3 = plot_device_workload_share(
        results,
        devices=net.servers,
        num_layers=model.num_layers,
        save_path=os.path.join(output_dir, "policy_device_workload_share.pdf"),
    )
    _emit(fig3)

    with open(
        os.path.join(output_dir, "policy_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({p: _json_safe(r) for p, r in results.items()}, f, indent=2)

    LOGGER.info("Wrote comparison artefacts to %s", output_dir)
    return results, output_dir
