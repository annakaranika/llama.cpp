import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graph_partitioning.compare_policies import layer_device_matrix
from graph_partitioning import Device, ModelSpec, Network

def main():
    # Find latest compare directory containing policy_results.json
    compare_files = glob.glob("sim_output/2026*/compare/policy_results.json")
    if not compare_files:
        compare_files = glob.glob("sim_output/compare/policy_results.json")
    if not compare_files:
        print("Error: Could not find policy_results.json in sim_output/*/compare/")
        return

    latest_json = max(compare_files, key=os.path.getmtime)
    output_dir = os.path.dirname(latest_json)
    print(f"Loading results from: {latest_json}")
    
    with open(latest_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Initialize model, network, and devices
    model = ModelSpec()
    num_layers = model.num_layers
    mem_per_layer = model.memory_usage_per_device(num_layers=[1])[0]

    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    servers = [
        Device("rpi-1", gflops=1.0, memory_gb=2.0),
        Device("rpi-2", gflops=3.0, memory_gb=1.0),
        Device("rpi-3", gflops=5.5, memory_gb=0.5),
        Device("rpi-4", gflops=0.4, memory_gb=1.0),
    ]

    device_names = [d.name for d in servers]
    capacities = [d.memory_gb for d in servers]

    # Extract memory allocations for each policy
    policies = ['single', 'pipeline', 'tensor', 'metis', 'alpa', 'sct']
    policy_labels = {
        'single': 'Single Node',
        'pipeline': 'Pipeline',
        'tensor': 'Tensor Parallel',
        'metis': 'METIS',
        'alpa': 'Alpa',
        'sct': 'SCT'
    }

    policy_mems = {}
    for p in policies:
        if p not in results:
            continue
        M = layer_device_matrix(results[p], num_layers, servers)
        mems = M.sum(axis=1) * mem_per_layer
        policy_mems[p] = mems

    # Set up beautiful color palette (Harmonious cool tones)
    colors = {
        'single': '#7a7a7a',      # Grey
        'pipeline': '#a6c4e0',    # Soft blue
        'tensor': '#9467bd',      # Purple
        'metis': '#71b171',       # Green
        'alpa': '#cab44e',        # Olive gold
        'sct': '#e38f45'          # Orange
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_groups = len(servers)
    n_policies = len(policy_mems)
    bar_width = 0.12
    x = np.arange(n_groups)

    # Plot grouped bars
    for idx, p in enumerate(policy_mems.keys()):
        offset = (idx - n_policies / 2.0 + 0.5) * bar_width
        mems = policy_mems[p]
        rects = ax.bar(
            x + offset, 
            mems, 
            bar_width, 
            label=policy_labels[p], 
            color=colors[p],
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add values on top of bars that violate memory capacity or for METIS
        for i, val in enumerate(mems):
            if val > capacities[i]:
                # Draw warning labels on top of violating bars
                ax.text(
                    x[i] + offset,
                    val + 0.05,
                    f"{val:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='red',
                    fontweight='bold',
                    rotation=90
                )
            elif p == 'metis' and val > 0:
                ax.text(
                    x[i] + offset,
                    val + 0.05,
                    f"{val:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='#2e692e',
                    fontweight='bold',
                    rotation=90
                )

    # Draw horizontal capacity limits for each device
    for i in range(n_groups):
        ax.hlines(
            y=capacities[i], 
            xmin=i - 0.45, 
            xmax=i + 0.45, 
            colors='red', 
            linestyles='--', 
            linewidth=1.8,
            label='Physical RAM Limit' if i == 0 else ""
        )
        ax.text(
            i - 0.4,
            capacities[i] + 0.05,
            f"Limit: {capacities[i]} GB",
            color='red',
            fontsize=9,
            fontweight='bold'
        )

    # Styling and aesthetics
    ax.set_xlabel('Device name (heterogeneous cluster)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Peak Memory Consumption (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Peak Memory Allocation vs. Physical Capacity by Partitioning Policy\n(Model Size: 2.45 GB, 23 Layers)', fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(device_names, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 2.8)
    
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Legend
    ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper right', shadow=True)
    
    plt.tight_layout()
    
    # Save files
    pdf_path = os.path.join(output_dir, "policy_memory_vs_capacity.pdf")
    png_path = os.path.join(output_dir, "policy_memory_vs_capacity.png")
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"Memory comparison plot saved to: {pdf_path}")
    print(f"Memory comparison plot saved to: {png_path}")

if __name__ == "__main__":
    main()
