"""Sweep model and device specs; emit per-policy latency line plots and 2D heatmaps.

Run with the graph_partitioning venv activated:

    source graph_partitioning/.venv/bin/activate
    python sweep_policies.py

Outputs land in ``sim_output/sweep/<timestamp>/``:
    sweep_num_devices.pdf      sweep_num_devices.csv
    sweep_model_size.pdf       sweep_model_size.csv
    sweep_bandwidth.pdf        sweep_bandwidth.csv
    sweep_seq_len.pdf          sweep_seq_len.csv
    sweep_gen_tokens.pdf       sweep_gen_tokens.csv
    heatmap_devices_x_bandwidth.pdf   .csv
    heatmap_seqlen_x_gentokens.pdf    .csv
"""

from graph_partitioning.sweep import run_all_sweeps


def main() -> None:
    output_dir = run_all_sweeps(
        device_counts=(2, 3, 4, 6, 8),
        model_scales=(0.5, 1.0, 2.0, 4.0),
        bandwidths_mbps=(10, 50, 100, 200, 500),
        seq_lens=(1, 16, 64, 256),
        gen_tokens_list=(1, 8, 32, 128),
    )
    print(f"\nSweep outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
