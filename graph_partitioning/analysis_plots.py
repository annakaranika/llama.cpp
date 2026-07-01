"""Reusable plotting functions for simulator analysis.

Used by simulator_analysis.ipynb and the standalone scripts in tests/root/.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from . import graph_plots
from .sweep import POLICY_COLOR, POLICY_MARKER, POLICY_RUNNERS


# ---------------------------------------------------------------------------
# Model workload capacity
# ---------------------------------------------------------------------------

TTFT_BUDGETS = [1.0, 3.0, 5.0, 10.0, 30.0]


def plot_capacity_preset(
    model_name: str,
    preset: str,
    n_devices: int,
    rows: List[Dict],
    save_path: Optional[str] = None,
    ttft_budgets: List[float] = TTFT_BUDGETS,
    tpt_target: float = 0.25,
) -> plt.Figure:
    """TTFT and TPT vs prompt length for one model/cluster combo."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    prompts = [r["prompt_len"] for r in rows]

    # Each line is the BEST policy at every prompt length, and TTFT vs TPT are
    # optimised INDEPENDENTLY — so the two lines can be different policies (e.g.
    # a distributed prefill for TTFT, but decode collapses to ~single-node for
    # TPT). Surface that: label the line with its winning policy and mark where
    # the best policy changes across prompt lengths.
    def _policy_label(metric, pols):
        uniq = [p for p in dict.fromkeys(pols) if p]  # ordered-unique, drop blanks
        if not uniq:
            return f"best {metric}"
        if len(uniq) == 1:
            return f"best {metric}: {uniq[0]}"
        return f"best {metric}: {' → '.join(uniq)} (varies)"

    def _mark_changes(ax, xs, ys, pols, color):
        prev = None
        for x, y, p in zip(xs, ys, pols):
            if p and p != prev:  # annotate first point + each transition
                ax.annotate(p, (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7, fontweight="bold", color=color)
                prev = p

    ttft_pols = [r.get("ttft_policy", "") for r in rows]
    tpt_pols = [r.get("tpt_policy", "") for r in rows]
    ttft_ys = [r["ttft_s"] for r in rows]
    tpt_ys = [r["tpt_s"] for r in rows]

    axes[0].plot(prompts, ttft_ys, marker="o", linewidth=2, color="#264b7b",
                 label=_policy_label("TTFT", ttft_pols))
    _mark_changes(axes[0], prompts, ttft_ys, ttft_pols, "#264b7b")
    for budget in ttft_budgets:
        axes[0].axhline(budget, linestyle=":", linewidth=1, alpha=0.6, color="green")
        axes[0].text(prompts[-1], budget, f"  {budget}s",
                     va="center", fontsize=8, color="green")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title(
        f"{model_name} on {preset} cluster (N={n_devices}) — TTFT vs prompt length",
        fontsize=11, fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(prompts, tpt_ys, marker="o", linewidth=2, color="#71b171",
                 label=_policy_label("TPT", tpt_pols))
    _mark_changes(axes[1], prompts, tpt_ys, tpt_pols, "#3f7f3f")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prompt length (tokens)")
    axes[1].set_ylabel("TPT (s/token)")
    axes[1].set_title("TPT vs prompt length", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].axhline(tpt_target, linestyle=":", linewidth=1, alpha=0.6, color="green")
    axes[1].text(prompts[-1], tpt_target,
                 f"  {tpt_target} s/tok\n  ({1/tpt_target:.0f} tok/s)",
                 va="center", fontsize=8, color="green")

    plt.tight_layout()
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Inference memory / latency timeline
# ---------------------------------------------------------------------------

def sample_points(context_len: int, gen_tokens: int, num_samples: int) -> List[int]:
    """KV-cache lengths sampling both prefill and decode phases."""
    prefill_n = max(2, num_samples // 2)
    decode_n  = max(2, num_samples - prefill_n)
    prefill = np.linspace(1, context_len, prefill_n, dtype=int).tolist()
    decode  = np.linspace(context_len + 1, context_len + gen_tokens,
                          decode_n, dtype=int).tolist()
    return sorted(set(prefill + decode))


_DEV_COLORS = ["#264b7b", "#a6c4e0", "#9467bd", "#e38f45", "#71b171", "#be6c6c"]


def plot_memory_timeline(
    sample_points_: List[int],
    timeline: List[dict],
    device_names: List[str],
    device_dram_gb: List[float],
    context_len: int,
    save_path: str,
    title: str,
) -> plt.Figure:
    """Grid of per-policy memory-usage traces."""
    policy_names = list(timeline[0].keys())
    cols = 3
    rows = (len(policy_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    all_mem = [
        timeline[i][p]["mem_gb"].get(d, 0.0)
        for i in range(len(timeline))
        for p in policy_names
        for d in device_names
    ]
    y_max = max(all_mem + list(device_dram_gb)) * 1.1

    for idx, policy in enumerate(policy_names):
        ax = axes[idx]
        for di, (name, cap) in enumerate(zip(device_names, device_dram_gb)):
            c = _DEV_COLORS[di % len(_DEV_COLORS)]
            ys = [s[policy]["mem_gb"].get(name, 0.0) for s in timeline]
            ax.plot(sample_points_, ys, color=c, marker="o", markersize=4,
                    linewidth=2, label=name)
            ax.axhline(cap, linestyle="--", linewidth=1, color=c, alpha=0.5)
        ax.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(policy, fontsize=11, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(policy_names):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles + [plt.Line2D([0], [0], color="gray", linestyle="--"),
                   plt.Line2D([0], [0], color="black", linestyle=":")],
        labels  + ["DRAM capacity", "prefill/decode boundary"],
        loc="lower center", ncol=min(len(device_names) + 2, 6),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_latency_timeline(
    sample_points_: List[int],
    timeline: List[dict],
    context_len: int,
    save_path: str,
    title: str,
    log_y: bool = True,
) -> plt.Figure:
    """One line per policy: per-decode-step latency vs KV-cache length."""
    policy_names = list(timeline[0].keys())
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for policy in policy_names:
        ax.plot(sample_points_, [s[policy]["latency_s"] for s in timeline],
                label=policy, color=POLICY_COLOR.get(policy, "#444"),
                marker=POLICY_MARKER.get(policy, "o"), markersize=5, linewidth=2)
    ax.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5,
               label="prefill/decode boundary")
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("per-decode-step latency (s)")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=min(len(policy_names) + 1, 4), frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_best_policy_timeline(
    sample_points_: List[int],
    timeline: List[dict],
    context_len: int,
    save_path: str,
    title: str,
) -> plt.Figure:
    """Lower envelope across policies; marker colour = winning policy."""
    policy_names = list(timeline[0].keys())
    winners, best_lats = [], []
    for step in timeline:
        valid = [(p, step[p]["latency_s"]) for p in policy_names
                 if not np.isnan(step[p]["latency_s"])]
        if valid:
            p, v = min(valid, key=lambda kv: kv[1])
            winners.append(p); best_lats.append(v)
        else:
            winners.append(None); best_lats.append(float("nan"))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for policy in policy_names:
        ax_top.plot(sample_points_,
                    [s[policy]["latency_s"] for s in timeline],
                    color="#bbbbbb", linewidth=1, alpha=0.7, zorder=1)
    ax_top.plot(sample_points_, best_lats, color="black", linewidth=1.5,
                linestyle="--", alpha=0.7, label="best per step", zorder=2)
    for policy in policy_names:
        xs = [x for x, w in zip(sample_points_, winners) if w == policy]
        ys = [y for y, w in zip(best_lats, winners) if w == policy]
        if xs:
            ax_top.scatter(xs, ys, color=POLICY_COLOR.get(policy, "#444"),
                           marker=POLICY_MARKER.get(policy, "o"),
                           s=70, zorder=3, label=policy,
                           edgecolors="black", linewidths=0.5)
    ax_top.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("best latency (s)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best", ncol=2, frameon=False, fontsize=9)

    policy_to_y = {p: i for i, p in enumerate(policy_names)}
    for x, w in zip(sample_points_, winners):
        if w:
            ax_bot.scatter(x, policy_to_y[w],
                           color=POLICY_COLOR.get(w, "#444"),
                           marker=POLICY_MARKER.get(w, "o"),
                           s=70, edgecolors="black", linewidths=0.5)
    ax_bot.axvline(context_len, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax_bot.set_yticks(list(policy_to_y.values()))
    ax_bot.set_yticklabels(policy_names)
    ax_bot.set_xlabel("KV cache length (tokens)")
    ax_bot.set_ylabel("winning policy")
    ax_bot.grid(True, alpha=0.3, axis="x")
    ax_bot.set_ylim(-0.5, len(policy_names) - 0.5)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if graph_plots.SAVE_FIGURES and save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
