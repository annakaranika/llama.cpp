"""For each model preset, find the smallest cluster size that hits a usable TTFT/TPT.

For each preset and each device-count N, runs every policy and reports the
*best* per-policy TTFT and TPT. Plots one panel per preset with x = N,
y = latency, two lines per policy (TTFT solid, TPT dashed).

Run:
    source graph_partitioning/.venv/bin/activate
    python find_min_cluster_size.py
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning import ModelSpec, Network, Device
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.sweep import (
    POLICY_COLOR,
    POLICY_RUNNERS,
    default_servers,
    run_all_policies_quiet,
)


MODEL_PRESETS = [
    ("tinyllama", ModelSpec(
        name="TinyLlama-1.1B",
        num_layers=23, num_heads=32, num_kv_heads=4,
        d_model=2048, d_k=64, d_ff=5632, vocab_size=32000,
    )),
    ("llama-7b", ModelSpec(
        name="Llama-7B",
        num_layers=32, num_heads=32, num_kv_heads=32,
        d_model=4096, d_k=128, d_ff=11008, vocab_size=32000,
    )),
    ("llama-13b", ModelSpec(
        name="Llama-13B",
        num_layers=40, num_heads=40, num_kv_heads=40,
        d_model=5120, d_k=128, d_ff=13824, vocab_size=32000,
    )),
]


# Workload + thresholds
CONTEXT_LEN = 256
GEN_TOKENS = 64
DEVICE_COUNTS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
TTFT_THRESHOLD_S = 1.0      # interactive chatbot bar for TTFT
TPT_THRESHOLD_S = 0.1       # 10 tokens/sec is "comfortable streaming"


def _build_net(servers):
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def run_one(model: ModelSpec, n_devices: int) -> Dict[str, Dict[str, float]]:
    servers = default_servers(n_devices)
    net_a = _build_net(servers)
    ttft = run_all_policies_quiet(model, net_a, CONTEXT_LEN, 1, kv_cache_seq_len=CONTEXT_LEN)
    net_b = _build_net(servers)
    mid = CONTEXT_LEN + GEN_TOKENS // 2
    tpt = run_all_policies_quiet(model, net_b, 1, 1, kv_cache_seq_len=mid)
    return {
        p: {"ttft_s": float(ttft.get(p, float("nan"))),
            "tpt_s":  float(tpt.get(p,  float("nan")))}
        for p in POLICY_RUNNERS
    }


def plot_preset(
    tag: str,
    model: ModelSpec,
    results: List[Dict[str, Dict[str, float]]],
    save_path: str,
) -> plt.Figure:
    """Two-panel figure for this preset: TTFT (top), TPT (bottom)."""
    fig, (ax_ttft, ax_tpt) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for policy in POLICY_RUNNERS:
        ttft_ys = [r[policy]["ttft_s"] for r in results]
        tpt_ys = [r[policy]["tpt_s"] for r in results]
        color = POLICY_COLOR.get(policy, "#444")
        ax_ttft.plot(DEVICE_COUNTS, ttft_ys, label=policy, color=color,
                     marker="o", markersize=5, linewidth=2)
        ax_tpt.plot(DEVICE_COUNTS, tpt_ys, label=policy, color=color,
                    marker="o", markersize=5, linewidth=2)

    ax_ttft.axhline(TTFT_THRESHOLD_S, color="green", linestyle=":",
                    label=f"TTFT target = {TTFT_THRESHOLD_S}s")
    ax_tpt.axhline(TPT_THRESHOLD_S, color="green", linestyle=":",
                   label=f"TPT target = {TPT_THRESHOLD_S}s/tok ({1/TPT_THRESHOLD_S:.0f} tok/s)")

    ax_ttft.set_ylabel("TTFT (s)")
    ax_ttft.set_yscale("log")
    ax_ttft.set_title(f"{model.name} — TTFT vs cluster size", fontsize=11, fontweight="bold")
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(loc="best", fontsize=8, frameon=False, ncol=2)

    ax_tpt.set_xlabel("number of RPi-class devices")
    ax_tpt.set_ylabel("TPT (s / decoded token)")
    ax_tpt.set_yscale("log")
    ax_tpt.set_title(f"{model.name} — TPT vs cluster size", fontsize=11, fontweight="bold")
    ax_tpt.grid(True, alpha=0.3)
    ax_tpt.legend(loc="best", fontsize=8, frameon=False, ncol=2)

    fig.suptitle(
        f"How many devices to make {model.name} usable?  "
        f"(prompt={CONTEXT_LEN}, decode={GEN_TOKENS})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def min_devices(results: List[Dict[str, Dict[str, float]]], metric: str, threshold: float):
    """Return per-policy smallest N (or None) that gets ``metric`` ≤ ``threshold``."""
    out: Dict[str, int] = {}
    for policy in POLICY_RUNNERS:
        for n, r in zip(DEVICE_COUNTS, results):
            val = r[policy][metric]
            if not np.isnan(val) and val <= threshold:
                out[policy] = n
                break
    return out


def main() -> None:
    out_dir = setup_output_dir("sim_output/min_cluster_size")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}\n"
          f"Sweeping device counts: {DEVICE_COUNTS}\n"
          f"TTFT target = {TTFT_THRESHOLD_S}s, TPT target = {TPT_THRESHOLD_S}s/tok\n")

    all_csv = os.path.join(out_dir, "all_results.csv")
    summary_csv = os.path.join(out_dir, "min_devices.csv")
    f_all = open(all_csv, "w", newline="", encoding="utf-8")
    w_all = csv.writer(f_all)
    w_all.writerow(["preset", "n_devices", "policy", "ttft_s", "tpt_s"])

    summary_rows = []

    for tag, model in MODEL_PRESETS:
        print(f"\n=== {model.name} ({tag}) ===")
        results = []
        for n in DEVICE_COUNTS:
            r = run_one(model, n)
            results.append(r)
            for p, vals in r.items():
                w_all.writerow([tag, n, p, f"{vals['ttft_s']:.4f}", f"{vals['tpt_s']:.4f}"])
            best_ttft = min(v["ttft_s"] for v in r.values() if not np.isnan(v["ttft_s"]))
            best_tpt  = min(v["tpt_s"]  for v in r.values() if not np.isnan(v["tpt_s"]))
            best_ttft_p = next(p for p, v in r.items() if v["ttft_s"] == best_ttft)
            best_tpt_p  = next(p for p, v in r.items() if v["tpt_s"]  == best_tpt)
            print(f"  N={n:>3d}  best TTFT={best_ttft:>9.2f}s ({best_ttft_p:>10s})  "
                  f"best TPT={best_tpt:>9.4f}s ({best_tpt_p:>10s})")

        min_ttft = min_devices(results, "ttft_s", TTFT_THRESHOLD_S)
        min_tpt  = min_devices(results, "tpt_s",  TPT_THRESHOLD_S)

        any_policy_reaches_ttft = min(min_ttft.values()) if min_ttft else None
        any_policy_reaches_tpt  = min(min_tpt.values())  if min_tpt  else None
        print(f"  → smallest N for TTFT ≤ {TTFT_THRESHOLD_S}s (any policy): "
              f"{any_policy_reaches_ttft}  detail: {min_ttft}")
        print(f"  → smallest N for TPT ≤ {TPT_THRESHOLD_S}s/tok (any policy): "
              f"{any_policy_reaches_tpt}   detail: {min_tpt}")

        summary_rows.append((tag, any_policy_reaches_ttft, any_policy_reaches_tpt))

        plot_preset(
            tag, model, results,
            save_path=os.path.join(out_dir, f"min_cluster_{tag}.pdf"),
        )

    f_all.close()
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["preset", "min_N_for_TTFT_target", "min_N_for_TPT_target"])
        for row in summary_rows:
            w.writerow(row)

    print(f"\nWrote per-preset plots and CSVs to {out_dir}")


if __name__ == "__main__":
    main()