"""Sweep model size; plot TTFT, TPT, and total latency per policy.

The simulator's per-policy ``total_latency_s = per_pass_time * gen_tokens``
treats every generated token as a fresh forward pass at ``seq_len``. That is
appropriate for *prefill* (one batched pass over the whole prompt) but not
for decode (one token with a growing KV cache). To approximate the standard
LLM metrics we run each policy twice per model scale:

  * with ``seq_len = context_len, gen_tokens = 1``  -> TTFT (prefill + 1 tok)
  * with ``seq_len = 1,            gen_tokens = 1``  -> per-decode-token cost (TPT)

End-to-end latency is then ``TTFT + (gen_tokens - 1) * TPT``.

Run with:
    source graph_partitioning/.venv/bin/activate
    python sweep_model_size.py
"""

from __future__ import annotations

import csv
import os

from graph_partitioning import Device, ModelSpec, Network
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.sweep import (
    plot_axis_lines,
    plot_device_memory_grid,
    run_all_policies_quiet,
    run_all_policies_with_memory,
    POLICY_RUNNERS,
)


# Three representative real-model presets (matching plot_inference_timeline).
# Sweeping these instead of scaled_model gives realistic d_k, GQA ratios, and
# vocab sizes — the things that drive KV-cache pressure in practice.
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


def _build_net(servers):
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )
    return net


def main() -> None:
    servers = [
        Device("rpi-1", gflops=1.0, memory_gb=2.0, swap_bandwidth_mbps=50.0),
        Device("rpi-2", gflops=3.0, memory_gb=1.0, swap_bandwidth_mbps=50.0),
        Device("rpi-3", gflops=5.5, memory_gb=2.0, swap_bandwidth_mbps=50.0),
        Device("rpi-4", gflops=0.4, memory_gb=1.0, swap_bandwidth_mbps=50.0),
    ]

    context_len = 256       # prompt length for TTFT
    gen_tokens = 64         # decode tokens for total latency

    out_dir = setup_output_dir("sim_output/sweep_model_size")
    os.makedirs(out_dir, exist_ok=True)

    preset_tags = [tag for tag, _ in MODEL_PRESETS]
    print(
        f"Sweeping presets: {preset_tags}\n"
        f"Devices: {[(d.name, d.gflops, d.memory_gb) for d in servers]}\n"
        f"Total cluster DRAM: {sum(d.memory_gb for d in servers):.1f} GB\n"
        f"context_len={context_len}, gen_tokens={gen_tokens}, swap_bw=50 Mbps/device\n"
        f"Output: {out_dir}\n"
    )

    header_cols = ["preset", "layers", "mem_GB"]
    header_cols += [f"{p}_TTFT" for p in POLICY_RUNNERS]
    header_cols += [f"{p}_TPT" for p in POLICY_RUNNERS]
    header_cols += [f"{p}_total" for p in POLICY_RUNNERS]

    ttft = {p: [] for p in POLICY_RUNNERS}
    tpt = {p: [] for p in POLICY_RUNNERS}
    total = {p: [] for p in POLICY_RUNNERS}
    model_sizes_gb = []
    # Per-preset snapshot of per-device memory (using the decode-phase
    # snapshot, which is the worst-case working set during inference).
    decode_phase_memory: list = []

    print(
        f"{'preset':>10} {'layers':>6} {'mem_GB':>7}   "
        + "  ".join(f"{p[:6]:>7s}/T" for p in POLICY_RUNNERS)
    )
    print("-" * 100)

    for tag, model in MODEL_PRESETS:
        mem_gb = model.total_memory_usage(seq_len=context_len)
        model_sizes_gb.append(mem_gb)

        # TTFT: one pass at the full context length. KV cache memory equals
        # context_len since we're about to process the whole prompt.
        net_a = _build_net(servers)
        ttft_lats = run_all_policies_quiet(
            model, net_a, context_len, 1, kv_cache_seq_len=context_len
        )

        # TPT: one *decode step* (single new token of compute), but the KV
        # cache sitting in device memory already covers the whole prompt
        # plus tokens generated so far. We use the midpoint of the decode
        # phase to approximate the average per-token swap pressure. Use the
        # memory-aware runner here so we also capture per-device footprint.
        net_b = _build_net(servers)
        mid_decode_ctx = context_len + gen_tokens // 2
        tpt_with_mem = run_all_policies_with_memory(
            model, net_b, 1, 1, kv_cache_seq_len=mid_decode_ctx
        )
        tpt_lats = {p: tpt_with_mem[p]["latency_s"] for p in POLICY_RUNNERS}
        decode_phase_memory.append(
            {p: tpt_with_mem[p] for p in POLICY_RUNNERS}
        )

        for p in POLICY_RUNNERS:
            ttft[p].append(ttft_lats.get(p, float("nan")))
            tpt[p].append(tpt_lats.get(p, float("nan")))
            total[p].append(ttft[p][-1] + (gen_tokens - 1) * tpt[p][-1])

        print(
            f"{tag:>10s} {model.num_layers:>6d} {mem_gb:>7.2f}   "
            + "  ".join(f"{ttft[p][-1]:>7.1f}/{tpt[p][-1]:>5.2f}" for p in POLICY_RUNNERS)
        )

    # X-axis labels: "tinyllama (2.4 GB)", etc.
    xlabels = [
        f"{tag} ({m:.1f} GB)"
        for (tag, _), m in zip(MODEL_PRESETS, model_sizes_gb)
    ]

    plot_axis_lines(
        "ttft",
        "model size (GB)",
        xlabels,
        ttft,
        save_path=os.path.join(out_dir, "ttft_vs_model_size.pdf"),
        log_y=True,
        y_label=f"TTFT (s) — prompt length = {context_len} tokens",
        metric_name="time-to-first-token",
    )
    plot_axis_lines(
        "tpt",
        "model size (GB)",
        xlabels,
        tpt,
        save_path=os.path.join(out_dir, "tpt_vs_model_size.pdf"),
        log_y=True,
        y_label="TPT (s/token)",
        metric_name="time-per-output-token",
    )
    plot_axis_lines(
        "total",
        "model size (GB)",
        xlabels,
        total,
        save_path=os.path.join(out_dir, "total_latency_vs_model_size.pdf"),
        log_y=True,
        y_label=(
            f"total latency (s) — prompt = {context_len} tok, "
            f"decode = {gen_tokens} tok"
        ),
        metric_name="total latency",
    )

    plot_device_memory_grid(
        decode_phase_memory,
        scales=model_sizes_gb,
        devices=servers,
        save_path=os.path.join(out_dir, "device_memory_vs_model_size.pdf"),
        x_label="model size (GB)",
    )

    # Per-device memory CSV (one row per (preset, policy, device))
    mem_csv_path = os.path.join(out_dir, "device_memory_vs_model_size.csv")
    with open(mem_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["preset", "model_size_gb", "policy", "device", "memory_gb"])
        for i, (tag, _) in enumerate(MODEL_PRESETS):
            for p in POLICY_RUNNERS:
                for d in servers:
                    mem = decode_phase_memory[i][p]["mem_gb"].get(d.name, 0.0)
                    w.writerow([tag, f"{model_sizes_gb[i]:.3f}", p, d.name, f"{mem:.4f}"])

    csv_path = os.path.join(out_dir, "latency_vs_model_size.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_cols)
        for i, (tag, model) in enumerate(MODEL_PRESETS):
            row = [tag, model.num_layers, f"{model_sizes_gb[i]:.3f}"]
            row += [f"{ttft[p][i]:.4f}" for p in POLICY_RUNNERS]
            row += [f"{tpt[p][i]:.4f}" for p in POLICY_RUNNERS]
            row += [f"{total[p][i]:.4f}" for p in POLICY_RUNNERS]
            w.writerow(row)

    print(f"\nWrote PDFs and CSV to {out_dir}")


if __name__ == "__main__":
    main()
