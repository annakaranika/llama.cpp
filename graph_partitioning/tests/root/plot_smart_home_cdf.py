"""CDFs of compute / DRAM / swap for the three smart-home presets."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from graph_partitioning.logging_utils import setup_output_dir
from graph_partitioning.smart_home_devices import (
    list_preset_names,
    preset_summary,
    smart_home_servers,
)


PRESET_COLORS = {"small": "#264b7b", "medium": "#71b171", "large": "#e38f45"}


def _stairs(values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Right-continuous empirical CDF."""
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    xs = np.concatenate(([arr[0]], np.repeat(arr, 2), [arr[-1] * 1.05]))
    ys = np.concatenate(([0.0], np.repeat(np.arange(1, n + 1) / n, 2)))
    return xs[: 2 * n + 1], ys[: 2 * n + 1]


def plot_metric(ax, get_value, *, resource, unit, log_x=False):
    for preset in list_preset_names():
        devs = smart_home_servers(preset)
        vals = [get_value(d) for d in devs]
        xs, ys = _stairs(vals)
        ax.plot(xs, ys, color=PRESET_COLORS[preset], linewidth=2.5,
                label=f"{preset} (N={len(devs)})")
        # Dot the actual sample points so you can see the discrete steps
        ax.scatter(np.sort(vals),
                   np.arange(1, len(vals) + 1) / len(vals),
                   color=PRESET_COLORS[preset], s=15, alpha=0.7)
    ax.set_xlabel(f"{resource} ({unit})")
    ax.set_ylabel("CDF")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)


def main() -> None:
    out_dir = setup_output_dir("sim_output/smart_home_cdf")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output: {out_dir}\n")
    for p in list_preset_names():
        s = preset_summary(p)
        print(f"  preset={p:>6s}  N={int(s['n_devices']):>3d}  "
              f"total compute={s['total_gflops']:>7.1f} GFLOPS  "
              f"total DRAM={s['total_dram_gb']:>6.1f} GB  "
              f"compute range=[{s['min_gflops']:>4.1f}, {s['max_gflops']:>5.1f}]  "
              f"DRAM range=[{s['min_dram_gb']:>4.2f}, {s['max_dram_gb']:>4.1f}]")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    plot_metric(axes[0], lambda d: d.gflops,
                resource="compute", unit="GFLOPS", log_x=True)
    plot_metric(axes[1], lambda d: d.memory_gb,
                resource="DRAM", unit="GB", log_x=True)
    plot_metric(axes[2], lambda d: d.swap_bandwidth_mbps,
                resource="swap bandwidth", unit="Mbps", log_x=True)
    fig.suptitle("Smart-home cluster CDFs (log-x): how many devices have ≤ x of each resource",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "smart_home_cdf.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
