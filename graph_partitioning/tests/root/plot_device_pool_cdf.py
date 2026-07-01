"""CDFs of the synthetic device pool used throughout the simulator.

Shows the distribution of compute (GFLOPS), DRAM (GB), and swap bandwidth
(Mbps) across the 10-device pool defined in ``default_servers``. Because
every ``find_min_cluster_size`` / ``voice_assistant_sizing`` /
``model_workload_capacity`` run cycles this pool, these CDFs describe the
hardware mix at every cluster size.

    source graph_partitioning/.venv/bin/activate
    python plot_device_pool_cdf.py
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning import Device
from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.sweep import default_servers


# Use a fixed "one full cycle" so each device-spec is counted once. Any larger
# N just repeats this distribution.
POOL_SIZE = 10


def _stairs(values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Right-continuous empirical CDF for a small set of values."""
    arr = np.sort(np.array(values, dtype=float))
    # Step function: at each unique x, F(x) = (# values <= x) / N
    n = len(arr)
    xs = np.concatenate(([arr[0]], np.repeat(arr, 2), [arr[-1] * 1.05]))
    ys = np.concatenate(([0.0], np.repeat(np.arange(1, n + 1) / n, 2)))
    # Trim to match length
    return xs[: 2 * n + 1], ys[: 2 * n + 1]


def plot_cdf(
    ax: plt.Axes,
    values: List[float],
    *,
    unit: str,
    color: str,
    title: str,
) -> None:
    xs, ys = _stairs(values)
    ax.plot(xs, ys, color=color, linewidth=2.5)
    ax.scatter(np.sort(values), np.arange(1, len(values) + 1) / len(values),
               color=color, s=40, zorder=5)
    median = float(np.median(values))
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    ax.axvline(median, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(median, 0.04, f" median = {median:g} {unit}",
            fontsize=8, va="bottom")
    ax.set_xlabel(f"value ({unit})")
    ax.set_ylabel("CDF")
    ax.set_title(
        f"{title}\nmin={min(values):g} • p10={p10:.2f} • "
        f"median={median:g} • p90={p90:.2f} • max={max(values):g} ({unit})",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)


def main() -> None:
    out_dir = setup_output_dir("sim_output/device_pool_cdf")
    os.makedirs(out_dir, exist_ok=True)

    pool: List[Device] = default_servers(POOL_SIZE)
    gflops = [d.gflops for d in pool]
    dram = [d.memory_gb for d in pool]
    swap = [d.swap_bandwidth_mbps for d in pool]

    print(f"Output: {out_dir}\n"
          f"Device pool (size={POOL_SIZE}):")
    for d in pool:
        print(f"  {d.name:>6s}  {d.gflops:>5.1f} GFLOPS  "
              f"{d.memory_gb:>4.1f} GB  swap={d.swap_bandwidth_mbps} Mbps")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_cdf(axes[0], gflops, unit="GFLOPS",
             color="#264b7b", title="compute capacity")
    plot_cdf(axes[1], dram, unit="GB",
             color="#71b171", title="DRAM capacity")
    plot_cdf(axes[2], swap, unit="Mbps",
             color="#e38f45", title="swap bandwidth")

    fig.suptitle(
        f"Device-pool CDFs ({POOL_SIZE}-device pool; larger N repeats this "
        f"distribution)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    pdf_path = os.path.join(out_dir, "device_pool_cdf.pdf")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
