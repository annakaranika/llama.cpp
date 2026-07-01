"""Cluster-sizing study for a private voice-assistant workload.

Workload assumptions:
  * Prompt: the spoken query plus any private RAG context the assistant
    pulls in to answer it (camera captions, sensor logs, calendar entries,
    chat history). We model this as a longish prompt (~2048 tokens) — short
    queries can still hit shorter prompts but the design point we care about
    is "how big a cluster do we need so that context-rich queries are
    answered in something close to natural speech timing?"
  * Response: a concise voice answer (~50 tokens). A voice assistant rarely
    monologues; it summarises.
  * Privacy is the reason we host the LLM locally, so we tolerate higher
    TTFT than a cloud-hosted UI would. Targets:
        - TTFT  ≤ 5.0 s   (the user spoke, the assistant takes a moment to
                            "consult" private context; tolerable)
        - TPT   ≤ 0.25 s/tok  (≥ 4 tok/s ≈ natural TTS speech rate of 150 WPM)
We deliberately ignore "total response latency" (TTFT + N·TPT for the full
response). For a streaming voice assistant the user starts hearing audio after
TTFT and the TPT just needs to keep up with TTS speech rate — those two
numbers, independently, are what matter.

For each model preset and each cluster size, runs every policy at the voice
workload and reports the *best* TTFT, TPT, and perceived-latency. Plots one
panel per preset with these three metrics.

    source graph_partitioning/.venv/bin/activate
    python voice_assistant_sizing.py
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning import Device, ModelSpec, Network
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


# Voice-assistant workload + thresholds. The prompt is dominated by private
# RAG context (recent camera captions, sensor logs, calendar / chat history)
# that gets piped into the LLM before the user's spoken query — so it's
# long, even though the user's actual utterance is short. The response stays
# short (a concise spoken answer).
PROMPT_TOKENS = 2048
RESPONSE_TOKENS = 50  # only used for the decode-midpoint KV-cache estimate
TTFT_TARGET_S = 5.0      # acceptable "consulting" pause for a private assistant
TPT_TARGET_S = 0.25      # ~4 tok/s ≈ natural TTS pace
DEVICE_COUNTS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def _build_net(servers):
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def run_one(model: ModelSpec, n_devices: int) -> Dict[str, Dict[str, float]]:
    """Run every policy on one (model, N) combo; return per-policy timings."""
    servers = default_servers(n_devices)

    # TTFT: prefill at the prompt length, KV cache = prompt length.
    net_a = _build_net(servers)
    ttft = run_all_policies_quiet(
        model, net_a, PROMPT_TOKENS, 1, kv_cache_seq_len=PROMPT_TOKENS,
    )

    # TPT: one decode step with KV cache at the midpoint of the response.
    net_b = _build_net(servers)
    mid_decode_ctx = PROMPT_TOKENS + RESPONSE_TOKENS // 2
    tpt = run_all_policies_quiet(
        model, net_b, 1, 1, kv_cache_seq_len=mid_decode_ctx,
    )

    out: Dict[str, Dict[str, float]] = {}
    for p in POLICY_RUNNERS:
        t_ttft = float(ttft.get(p, float("nan")))
        t_tpt = float(tpt.get(p, float("nan")))
        out[p] = {"ttft_s": t_ttft, "tpt_s": t_tpt}
    return out


def plot_preset(model: ModelSpec, results: List[Dict], save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)
    metrics = [
        ("ttft_s", "TTFT (s)", TTFT_TARGET_S,
         f"TTFT target = {TTFT_TARGET_S}s"),
        ("tpt_s", "TPT (s/token)", TPT_TARGET_S,
         f"TPT target = {TPT_TARGET_S}s/tok (≈{1/TPT_TARGET_S:.0f} tok/s)"),
    ]
    for ax, (metric, ylabel, threshold, threshold_label) in zip(axes, metrics):
        for policy in POLICY_RUNNERS:
            ys = [r[policy][metric] for r in results]
            ax.plot(
                DEVICE_COUNTS, ys, label=policy,
                color=POLICY_COLOR.get(policy, "#444"),
                marker="o", markersize=5, linewidth=2,
            )
        ax.axhline(threshold, color="green", linestyle=":", label=threshold_label)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)

    axes[-1].set_xlabel("number of RPi-class devices")
    fig.suptitle(
        f"Voice-assistant sizing — {model.name}\n"
        f"prompt={PROMPT_TOKENS} tok, response={RESPONSE_TOKENS} tok",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def min_devices(results: List[Dict], metric: str, threshold: float) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for policy in POLICY_RUNNERS:
        for n, r in zip(DEVICE_COUNTS, results):
            v = r[policy][metric]
            if not np.isnan(v) and v <= threshold:
                out[policy] = n
                break
    return out


def main() -> None:
    out_dir = setup_output_dir("sim_output/voice_assistant_sizing")
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"Output: {out_dir}\n"
        f"Workload: prompt={PROMPT_TOKENS} tok, response={RESPONSE_TOKENS} tok\n"
        f"Targets: TTFT ≤ {TTFT_TARGET_S}s, TPT ≤ {TPT_TARGET_S}s/tok\n"
        f"Sweeping N = {DEVICE_COUNTS}\n"
    )

    all_csv = open(os.path.join(out_dir, "voice_results.csv"), "w",
                   newline="", encoding="utf-8")
    w_all = csv.writer(all_csv)
    w_all.writerow(["preset", "n_devices", "policy", "ttft_s", "tpt_s"])

    summary_rows = []
    for tag, model in MODEL_PRESETS:
        print(f"\n=== {model.name} ===")
        results = []
        for n in DEVICE_COUNTS:
            r = run_one(model, n)
            results.append(r)
            for p, v in r.items():
                w_all.writerow([
                    tag, n, p,
                    f"{v['ttft_s']:.4f}", f"{v['tpt_s']:.4f}",
                ])
            best = lambda key: min(  # noqa: E731
                (v[key] for v in r.values() if not np.isnan(v[key])),
                default=float("nan"),
            )
            print(
                f"  N={n:>3d}  best TTFT={best('ttft_s'):>8.2f}s   "
                f"best TPT={best('tpt_s'):>8.4f}s"
            )

        m_ttft = min_devices(results, "ttft_s", TTFT_TARGET_S)
        m_tpt = min_devices(results, "tpt_s", TPT_TARGET_S)

        def best_of(d):
            return min(d.values()) if d else None

        print(f"  → min N to hit TTFT target (any policy):  "
              f"{best_of(m_ttft)}  {m_ttft}")
        print(f"  → min N to hit TPT target (any policy):   "
              f"{best_of(m_tpt)}  {m_tpt}")

        summary_rows.append({
            "preset": tag,
            "min_N_TTFT": best_of(m_ttft),
            "min_N_TPT": best_of(m_tpt),
            "ttft_winners": m_ttft,
            "tpt_winners": m_tpt,
        })
        plot_preset(model, results,
                    save_path=os.path.join(out_dir, f"voice_{tag}.pdf"))

    all_csv.close()

    with open(os.path.join(out_dir, "voice_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["preset", "min_N_TTFT_target", "min_N_TPT_target"])
        for s in summary_rows:
            w.writerow([s["preset"], s["min_N_TTFT"], s["min_N_TPT"]])

    print(f"\nWrote per-preset plots, full CSV, and summary CSV to {out_dir}")


if __name__ == "__main__":
    main()