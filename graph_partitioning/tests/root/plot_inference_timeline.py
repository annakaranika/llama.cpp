"""Plot per-device memory usage over the course of one inference, for a fixed model.

The x-axis is "tokens in the KV cache" — equivalently, time since the request
arrived if we charge each prefill token and each decode token as one step.
A dashed vertical line marks the prefill/decode boundary; dashed horizontal
lines mark each device's DRAM capacity, so swap regimes are visible at a
glance.

Pass ``--scale`` to pick the model size (default 1.0 ≈ TinyLlama),
``--context``, ``--gen`` to control the workload.

    source graph_partitioning/.venv/bin/activate
    python plot_inference_timeline.py --scale 1.5 --context 1024 --gen 128
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning import Device, ModelSpec, Network
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.sweep import (
    POLICY_COLOR,
    POLICY_MARKER,
    POLICY_RUNNERS,
    run_all_policies_with_memory,
    scaled_model,
)


# Realistic LLM presets. Real models have larger d_k and much less aggressive
# (or no) GQA, so KV cache memory is meaningful at long contexts.
MODEL_PRESETS = {
    "tinyllama": ModelSpec(
        name="TinyLlama-1.1B",
        num_layers=23, num_heads=32, num_kv_heads=4,
        d_model=2048, d_k=64, d_ff=5632, vocab_size=32000,
    ),
    "llama-7b": ModelSpec(
        name="Llama-7B",
        num_layers=32, num_heads=32, num_kv_heads=32,  # no GQA
        d_model=4096, d_k=128, d_ff=11008, vocab_size=32000,
    ),
    "llama-13b": ModelSpec(
        name="Llama-13B",
        num_layers=40, num_heads=40, num_kv_heads=40,
        d_model=5120, d_k=128, d_ff=13824, vocab_size=32000,
    ),
}


def _build_net(servers):
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def _sample_points(context_len: int, gen_tokens: int, num_samples: int) -> List[int]:
    """Pick KV-cache lengths that sample both phases.

    Half the samples go to prefill (1..context_len) and half to decode
    (context_len+1 .. context_len+gen_tokens). Dedupe and sort.
    """
    prefill_n = max(2, num_samples // 2)
    decode_n = max(2, num_samples - prefill_n)
    prefill = np.linspace(1, context_len, prefill_n, dtype=int).tolist()
    decode = np.linspace(
        context_len + 1, context_len + gen_tokens, decode_n, dtype=int
    ).tolist()
    return sorted(set(prefill + decode))


def plot_timeline(
    sample_points: List[int],
    timeline: List[dict],
    devices: List[Device],
    context_len: int,
    save_path: str,
    model_label: str,
    model_mem_gb: float,
) -> plt.Figure:
    policy_names = list(timeline[0].keys())
    n = len(policy_names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 3.6 * rows), sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    dev_color = ["#264b7b", "#a6c4e0", "#9467bd", "#e38f45", "#71b171", "#be6c6c"]

    # Global y-axis range: every subplot uses the same scale so cross-policy
    # comparisons are honest. Include device DRAM capacities so dashed
    # capacity lines stay on-axis even for policies that never approach them.
    all_mem_vals = [
        step[policy]["mem_gb"].get(dev.name, 0.0)
        for step in timeline
        for policy in policy_names
        for dev in devices
    ]
    y_max = max(all_mem_vals + [d.memory_gb for d in devices]) * 1.1
    y_min = 0.0

    for ax_idx, policy in enumerate(policy_names):
        ax = axes[ax_idx]
        for d_idx, dev in enumerate(devices):
            ys = [step[policy]["mem_gb"].get(dev.name, 0.0) for step in timeline]
            ax.plot(
                sample_points,
                ys,
                label=dev.name,
                color=dev_color[d_idx % len(dev_color)],
                marker="o",
                markersize=4,
                linewidth=2,
            )
            ax.axhline(
                dev.memory_gb,
                linestyle="--",
                linewidth=1,
                color=dev_color[d_idx % len(dev_color)],
                alpha=0.5,
            )
        # Prefill/decode boundary
        ax.axvline(
            context_len,
            color="black",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )
        ax.set_title(policy, fontsize=11, fontweight="bold")
        ax.set_xlabel("KV cache length (tokens)")
        ax.set_ylabel("memory used (GB)")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(policy_names):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles + [
            plt.Line2D([0], [0], color="gray", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle=":"),
        ],
        labels + ["device DRAM capacity", "prefill / decode boundary"],
        loc="lower center",
        ncol=min(len(devices) + 2, 6),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        f"Per-device memory over inference timeline — "
        f"{model_label} ({model_mem_gb:.2f} GB model)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_latency_timeline(
    sample_points: List[int],
    timeline: List[dict],
    context_len: int,
    save_path: str,
    model_label: str,
    log_y: bool = True,
) -> plt.Figure:
    """One axis, one line per policy: per-decode-step latency vs KV length."""
    policy_names = list(timeline[0].keys())
    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    for policy in policy_names:
        ys = [step[policy]["latency_s"] for step in timeline]
        ax.plot(
            sample_points,
            ys,
            label=policy,
            color=POLICY_COLOR.get(policy, "#444"),
            marker=POLICY_MARKER.get(policy, "o"),
            markersize=5,
            linewidth=2,
        )
    ax.axvline(
        context_len, color="black", linestyle=":", linewidth=1, alpha=0.5,
        label="prefill / decode boundary",
    )
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("per-decode-step latency (s)")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Per-decode-step latency over inference timeline — {model_label}",
        fontsize=12, fontweight="bold",
    )
    # Legend sits between suptitle and the axes, like the memory plot.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=min(len(policy_names) + 1, 4),
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_best_policy_timeline(
    sample_points: List[int],
    timeline: List[dict],
    context_len: int,
    save_path: str,
    model_label: str,
) -> plt.Figure:
    """Lower envelope across policies: per-step minimum latency, marker colour = winning policy.

    Two panels:
      * top: best per-step latency (log y), with each marker coloured by the
        policy that achieved it; a faint grey line per policy is shown in the
        background so you can see how much the winner beats the runners-up.
      * bottom: categorical step plot showing the winning policy at each
        sample point — makes regime changes (e.g. policy X wins during
        prefill, policy Y wins during decode) immediately visible.
    """
    policy_names = list(timeline[0].keys())
    # Find winner per step
    winners = []
    best_lats = []
    for step in timeline:
        valid = [(p, step[p]["latency_s"]) for p in policy_names
                 if not np.isnan(step[p]["latency_s"])]
        if not valid:
            winners.append(None)
            best_lats.append(float("nan"))
            continue
        p, v = min(valid, key=lambda kv: kv[1])
        winners.append(p)
        best_lats.append(v)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Background: every policy in faint grey
    for policy in policy_names:
        ys = [step[policy]["latency_s"] for step in timeline]
        ax_top.plot(sample_points, ys, color="#bbbbbb", linewidth=1,
                    alpha=0.7, zorder=1)

    # Lower envelope (best per step) as one line
    ax_top.plot(sample_points, best_lats, color="black", linewidth=1.5,
                linestyle="--", alpha=0.7, label="best per step", zorder=2)

    # Markers coloured by winning policy
    for policy in policy_names:
        xs = [x for x, w in zip(sample_points, winners) if w == policy]
        ys = [y for y, w in zip(best_lats, winners) if w == policy]
        if not xs:
            continue
        ax_top.scatter(xs, ys,
                       color=POLICY_COLOR.get(policy, "#444"),
                       marker=POLICY_MARKER.get(policy, "o"),
                       s=70, zorder=3, label=policy, edgecolors="black",
                       linewidths=0.5)

    ax_top.axvline(context_len, color="black", linestyle=":", linewidth=1,
                   alpha=0.5)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("best per-decode-step latency (s)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best", ncol=2, frameon=False, fontsize=9)

    # Bottom: categorical "which policy wins" strip
    policy_to_y = {p: i for i, p in enumerate(policy_names)}
    for x, w in zip(sample_points, winners):
        if w is None:
            continue
        ax_bot.scatter(x, policy_to_y[w],
                       color=POLICY_COLOR.get(w, "#444"),
                       marker=POLICY_MARKER.get(w, "o"),
                       s=70, edgecolors="black", linewidths=0.5)
    ax_bot.axvline(context_len, color="black", linestyle=":", linewidth=1,
                   alpha=0.5)
    ax_bot.set_yticks(list(policy_to_y.values()))
    ax_bot.set_yticklabels(policy_names)
    ax_bot.set_xlabel("KV cache length (tokens)")
    ax_bot.set_ylabel("winning policy")
    ax_bot.grid(True, alpha=0.3, axis="x")
    ax_bot.set_ylim(-0.5, len(policy_names) - 0.5)

    fig.suptitle(
        f"Best policy along the inference timeline — {model_label}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_combined_timeline(
    sample_points: List[int],
    timeline: List[dict],
    devices: List[Device],
    context_len: int,
    save_path: str,
    model_label: str,
    model_mem_gb: float,
) -> plt.Figure:
    """One figure: per-policy memory grid (top) + shared latency lines (bottom).

    Both panels share the KV-cache-length x-axis so memory pressure and the
    resulting latency are readable at the same time-point.
    """
    policy_names = list(timeline[0].keys())
    n = len(policy_names)
    cols = 3
    mem_rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5.0 * cols, 3.4 * mem_rows + 4.0))
    # Top: memory grid; bottom: single-axis latency
    gs = fig.add_gridspec(
        mem_rows + 1, cols,
        height_ratios=[3] * mem_rows + [4],
        hspace=0.45, wspace=0.30,
    )

    dev_color = ["#264b7b", "#a6c4e0", "#9467bd", "#e38f45", "#71b171", "#be6c6c"]

    # Shared memory y-limit across all policy panels
    all_mem_vals = [
        step[policy]["mem_gb"].get(dev.name, 0.0)
        for step in timeline
        for policy in policy_names
        for dev in devices
    ]
    y_max_mem = max(all_mem_vals + [d.memory_gb for d in devices]) * 1.1

    # Per-policy memory subplots
    first_mem_ax = None
    for ax_idx, policy in enumerate(policy_names):
        row = ax_idx // cols
        col = ax_idx % cols
        ax = fig.add_subplot(
            gs[row, col],
            sharey=first_mem_ax if first_mem_ax is not None else None,
        )
        if first_mem_ax is None:
            first_mem_ax = ax
        for d_idx, dev in enumerate(devices):
            ys = [step[policy]["mem_gb"].get(dev.name, 0.0) for step in timeline]
            ax.plot(
                sample_points, ys,
                color=dev_color[d_idx % len(dev_color)],
                marker="o", markersize=3, linewidth=1.5,
                label=dev.name if ax_idx == 0 else None,
            )
            ax.axhline(
                dev.memory_gb,
                linestyle="--", linewidth=1,
                color=dev_color[d_idx % len(dev_color)], alpha=0.5,
            )
        ax.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(policy, fontsize=10, fontweight="bold")
        ax.set_ylim(0, y_max_mem)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("mem (GB)")

    # Bottom: latency lines (all policies)
    ax_lat = fig.add_subplot(gs[mem_rows, :])
    for policy in policy_names:
        ys = [step[policy]["latency_s"] for step in timeline]
        ax_lat.plot(
            sample_points, ys,
            label=policy,
            color=POLICY_COLOR.get(policy, "#444"),
            marker=POLICY_MARKER.get(policy, "o"),
            markersize=5, linewidth=2,
        )
    ax_lat.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax_lat.set_xlabel("KV cache length (tokens)")
    ax_lat.set_ylabel("per-decode-step latency (s, log)")
    ax_lat.set_yscale("log")
    ax_lat.set_title("latency over inference timeline", fontsize=11, fontweight="bold")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="best", ncol=3, frameon=False, fontsize=9)

    # Shared device legend for the memory panels (first_mem_ax always exists
    # because policy_names is non-empty for any real timeline).
    assert first_mem_ax is not None
    handles, labels = first_mem_ax.get_legend_handles_labels()
    fig.legend(
        handles + [
            plt.Line2D([0], [0], color="gray", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle=":"),
        ],
        labels + ["device DRAM capacity", "prefill / decode boundary"],
        loc="upper center",
        ncol=min(len(devices) + 2, 6),
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.suptitle(
        f"Inference timeline — {model_label} ({model_mem_gb:.2f} GB model)",
        fontsize=14, fontweight="bold", y=1.04,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default=None,
        choices=sorted(MODEL_PRESETS.keys()),
        help="model preset (overrides --scale)",
    )
    ap.add_argument("--scale", type=float, default=1.0,
                    help="model scale factor (only used when --model is not given)")
    ap.add_argument("--context", type=int, default=256, help="prompt length")
    ap.add_argument("--gen", type=int, default=128, help="generated tokens")
    ap.add_argument("--samples", type=int, default=24,
                    help="total sample points (split across prefill/decode)")
    args = ap.parse_args()

    servers = [
        Device("rpi-1", gflops=1.0, memory_gb=2.0, swap_bandwidth_mbps=50.0),
        Device("rpi-2", gflops=3.0, memory_gb=1.0, swap_bandwidth_mbps=50.0),
        Device("rpi-3", gflops=5.5, memory_gb=2.0, swap_bandwidth_mbps=50.0),
        Device("rpi-4", gflops=0.4, memory_gb=1.0, swap_bandwidth_mbps=50.0),
    ]

    if args.model:
        model = MODEL_PRESETS[args.model]
        model_tag = args.model
    else:
        model = scaled_model(args.scale)
        model_tag = f"scaled_{args.scale:.2f}"
    sample_points = _sample_points(args.context, args.gen, args.samples)

    out_dir = setup_output_dir("sim_output/inference_timeline")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Model: {model.name} ({model_tag})  "
        f"layers={model.num_layers}  d_model={model.d_model}  "
        f"d_k={model.d_k}  kv_heads={model.num_kv_heads}  "
        f"kv_bytes_per_head={model.kv_bytes_per_head}\n"
        f"Workload: context={args.context} tok, decode={args.gen} tok\n"
        f"Sampling {len(sample_points)} points: {sample_points}\n"
        f"Output: {out_dir}\n"
    )

    timeline = []
    for kv_len in sample_points:
        net = _build_net(servers)
        # seq_len=1 for compute (single forward pass), kv_cache_seq_len=kv_len
        # for memory pressure at this point in the timeline.
        result = run_all_policies_with_memory(
            model, net, 1, 1, kv_cache_seq_len=kv_len
        )
        timeline.append(result)

    label = model.name if args.model else model_tag
    model_mem_gb = model.total_memory_usage(seq_len=args.context)

    plot_timeline(
        sample_points, timeline,
        devices=servers, context_len=args.context,
        save_path=os.path.join(out_dir, f"memory_timeline_{model_tag}.pdf"),
        model_label=label, model_mem_gb=model_mem_gb,
    )
    plot_latency_timeline(
        sample_points, timeline,
        context_len=args.context,
        save_path=os.path.join(out_dir, f"latency_timeline_{model_tag}.pdf"),
        model_label=label,
    )
    plot_combined_timeline(
        sample_points, timeline,
        devices=servers, context_len=args.context,
        save_path=os.path.join(out_dir, f"timeline_combined_{model_tag}.pdf"),
        model_label=label, model_mem_gb=model_mem_gb,
    )
    plot_best_policy_timeline(
        sample_points, timeline,
        context_len=args.context,
        save_path=os.path.join(out_dir, f"best_policy_timeline_{model_tag}.pdf"),
        model_label=label,
    )

    mem_csv_path = os.path.join(out_dir, "memory_timeline.csv")
    with open(mem_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kv_len", "policy", "device", "memory_gb"])
        for i, kv in enumerate(sample_points):
            for p in POLICY_RUNNERS:
                for d in servers:
                    mem = timeline[i][p]["mem_gb"].get(d.name, 0.0)
                    w.writerow([kv, p, d.name, f"{mem:.4f}"])

    lat_csv_path = os.path.join(out_dir, "latency_timeline.csv")
    with open(lat_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kv_len", "policy", "latency_s"])
        for i, kv in enumerate(sample_points):
            for p in POLICY_RUNNERS:
                w.writerow([kv, p, f"{timeline[i][p]['latency_s']:.6f}"])

    print(f"Wrote memory  plot to {out_dir}/memory_timeline_{model_tag}.pdf")
    print(f"Wrote latency plot to {out_dir}/latency_timeline_{model_tag}.pdf")
    print(f"Wrote best-policy plot to {out_dir}/best_policy_timeline_{model_tag}.pdf")
    print(f"Wrote combined    to {out_dir}/timeline_combined_{model_tag}.pdf")
    print(f"Wrote memory CSV  to {mem_csv_path}")
    print(f"Wrote latency CSV to {lat_csv_path}")


if __name__ == "__main__":
    main()