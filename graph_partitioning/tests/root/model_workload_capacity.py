"""For each model preset on its best-effort cluster, find the largest prompt
and the per-token decode rate the cluster can actually deliver.

We pick the cluster size that wins the TPT race for each preset (from the
earlier voice-assistant sweep), then sweep prompt length to find:
  * the max prompt that fits each TTFT budget (1s, 3s, 5s, 10s, 30s)
  * the steady-state TPT at that prompt — independent of prompt for decode,
    but we re-measure at the same KV-cache snapshot for consistency.

Response length is implicit: once you know TPT, "max response in budget B" is
just ``(B - TTFT) / TPT``. We report that for a 10-second-of-speech budget
(~40 tokens at 4 tok/s).

    source graph_partitioning/.venv/bin/activate
    python model_workload_capacity.py
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning import Device, ModelSpec, Network
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.smart_home_devices import smart_home_servers
from graph_partitioning.sweep import (
    POLICY_COLOR,
    POLICY_RUNNERS,
    run_all_policies_quiet,
)


# Each entry: (tag, ModelSpec, smart-home preset name).
# We pair each model with the cluster size most plausibly required to host
# it in a private home: a tiny model fits on a small smart-home setup,
# while bigger ones need an enthusiast / heavy-automation cluster.
MODEL_PRESETS = [
    ("tinyllama", ModelSpec(
        name="TinyLlama-1.1B",
        num_layers=23, num_heads=32, num_kv_heads=4,
        d_model=2048, d_k=64, d_ff=5632, vocab_size=32000,
    ), "small"),    # 4 smart-home devices
    ("llama-7b", ModelSpec(
        name="Llama-7B",
        num_layers=32, num_heads=32, num_kv_heads=32,
        d_model=4096, d_k=128, d_ff=11008, vocab_size=32000,
    ), "medium"),   # 32 smart-home devices
    ("llama-13b", ModelSpec(
        name="Llama-13B",
        num_layers=40, num_heads=40, num_kv_heads=40,
        d_model=5120, d_k=128, d_ff=13824, vocab_size=32000,
    ), "large"),    # 48 smart-home devices
]


PROMPT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TTFT_BUDGETS = [1.0, 3.0, 5.0, 10.0, 30.0]
SPEECH_BUDGET_S = 10.0  # 10 seconds of streamed answer is a long voice reply


def _build_net(servers):
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def best_ttft_tpt(model: ModelSpec, preset: str, prompt_len: int
                  ) -> Tuple[float, str, float, str]:
    """Return (best_ttft, best_ttft_policy, best_tpt, best_tpt_policy)."""
    servers = smart_home_servers(preset)
    net_a = _build_net(servers)
    ttft = run_all_policies_quiet(
        model, net_a, prompt_len, 1, kv_cache_seq_len=prompt_len,
    )
    net_b = _build_net(servers)
    # Use a representative mid-decode KV size (prompt + 25 tokens).
    tpt = run_all_policies_quiet(
        model, net_b, 1, 1, kv_cache_seq_len=prompt_len + 25,
    )

    def _best(d):
        valid = [(p, v) for p, v in d.items() if not np.isnan(v)]
        if not valid:
            return float("nan"), ""
        p, v = min(valid, key=lambda kv: kv[1])
        return v, p

    best_ttft_s, best_ttft_p = _best(ttft)
    best_tpt_s, best_tpt_p = _best(tpt)
    return best_ttft_s, best_ttft_p, best_tpt_s, best_tpt_p


def plot_preset(model: ModelSpec, preset: str, n_devices: int,
                rows: List[Dict], save_path: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    ttft_y = [r["ttft_s"] for r in rows]
    tpt_y = [r["tpt_s"] for r in rows]
    prompts = [r["prompt_len"] for r in rows]

    axes[0].plot(prompts, ttft_y, marker="o", linewidth=2, color="#264b7b")
    for budget in TTFT_BUDGETS:
        axes[0].axhline(budget, linestyle=":", linewidth=1, alpha=0.6,
                        color="green")
        axes[0].text(prompts[-1], budget, f"  {budget}s", va="center",
                     fontsize=8, color="green")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title(
        f"{model.name} on {preset} smart-home cluster "
        f"(N={n_devices} devices) — TTFT vs prompt length",
        fontsize=11, fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(prompts, tpt_y, marker="o", linewidth=2, color="#71b171")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prompt length (tokens)")
    axes[1].set_ylabel("TPT (s/token)")
    axes[1].set_title("TPT vs prompt length", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.25, linestyle=":", linewidth=1, alpha=0.6, color="green")
    axes[1].text(prompts[-1], 0.25, "  0.25 s/tok\n  (4 tok/s)", va="center",
                 fontsize=8, color="green")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    out_dir = setup_output_dir("sim_output/model_workload_capacity")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output: {out_dir}\n"
          f"Sweeping prompt lengths {PROMPT_LENGTHS}\n"
          f"TTFT budgets: {TTFT_BUDGETS}\n"
          f"Speech budget for max-response calculation: {SPEECH_BUDGET_S}s\n")

    summary_csv = open(os.path.join(out_dir, "capacity_summary.csv"), "w",
                       newline="", encoding="utf-8")
    sw = csv.writer(summary_csv)
    sw.writerow(["preset", "smart_home", "n_devices", "ttft_budget_s",
                 "max_prompt_tokens", "tpt_s_at_that_prompt",
                 "max_response_in_10s_speech"])

    for tag, model, preset in MODEL_PRESETS:
        servers = smart_home_servers(preset)
        n = len(servers)
        print(f"\n=== {model.name} on {preset} smart-home cluster "
              f"(N={n} devices) ===")
        rows = []
        print(f"  {'prompt':>6}  {'TTFT(s)':>9} ({'policy':>10})  "
              f"{'TPT(s)':>8} ({'policy':>10})")
        for prompt in PROMPT_LENGTHS:
            tt, tt_p, tp, tp_p = best_ttft_tpt(model, preset, prompt)
            rows.append({"prompt_len": prompt, "ttft_s": tt, "tpt_s": tp,
                         "ttft_policy": tt_p, "tpt_policy": tp_p})
            print(f"  {prompt:>6d}  {tt:>9.2f} ({tt_p:>10s})  "
                  f"{tp:>8.4f} ({tp_p:>10s})")

        for budget in TTFT_BUDGETS:
            fit = [r for r in rows if not np.isnan(r["ttft_s"])
                   and r["ttft_s"] <= budget]
            if not fit:
                print(f"  → TTFT ≤ {budget:>5.1f}s : NO prompt length fits")
                sw.writerow([tag, preset, n, budget, None, None, None])
                continue
            best = max(fit, key=lambda r: r["prompt_len"])
            tpt_here = best["tpt_s"]
            max_resp = int(max(0, (SPEECH_BUDGET_S - best["ttft_s"]) / tpt_here)) \
                       if tpt_here > 0 else 0
            print(f"  → TTFT ≤ {budget:>5.1f}s : "
                  f"max prompt {best['prompt_len']:>5d} tok  "
                  f"(TPT {tpt_here:.3f}s/tok ≈ {1/tpt_here:.1f} tok/s)  "
                  f"→ max response in {SPEECH_BUDGET_S:.0f}s speech: "
                  f"{max_resp} tok")
            sw.writerow([tag, preset, n, budget, best["prompt_len"],
                         f"{tpt_here:.4f}", max_resp])

        plot_preset(model, preset, n, rows,
                    save_path=os.path.join(out_dir, f"capacity_{tag}.pdf"))

    summary_csv.close()
    print(f"\nWrote per-preset plots and summary CSV to {out_dir}")


if __name__ == "__main__":
    main()
