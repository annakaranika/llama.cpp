"""Functions to visualize computation graph partitions and device assignments."""

from typing import Dict, List, Tuple, Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import patches
import networkx as nx
import numpy as np

from .common import AttentionSplitStrategy, Device, FFNSplitStrategy
from .logging_utils import setup_logger
from .model import ModelSpec
from .network import Network

LOGGER = setup_logger(name=__name__)
_output_dir: Optional[str] = None

# The per-policy diagnostic figures (partition graph, device computation
# breakdown, resource utilisation) are produced as a side effect of *every*
# simulate_* call, so display and disk-writing are gated independently:
#   SAVE_FIGURES — write the PDF to disk.
#   SHOW_FIGURES — display() the figure inline (notebooks).
# SHOW defaults off because bulk loops (sweeps, the §10 prediction loop) call
# simulate_* dozens of times; a visualisation cell turns it on explicitly via
# set_show_figures(True). Either way the figure is always closed, never leaked.
SAVE_FIGURES: bool = True
SHOW_FIGURES: bool = False


def set_save_figures(enabled: bool) -> None:
    """Enable/disable writing the diagnostic partition figures to disk."""
    global SAVE_FIGURES  # pylint: disable=global-statement
    SAVE_FIGURES = enabled


def set_show_figures(enabled: bool) -> None:
    """Enable/disable displaying the diagnostic partition figures inline."""
    global SHOW_FIGURES  # pylint: disable=global-statement
    SHOW_FIGURES = enabled


def _emit_figure(fig) -> None:
    """Display the figure inline when SHOW_FIGURES is on (and IPython is
    available), then always close it so nothing leaks into later cells."""
    if SHOW_FIGURES:
        try:
            from IPython.display import display  # noqa: PLC0415
            display(fig)
        except Exception:  # pragma: no cover - not in a notebook
            pass
    plt.close(fig)


def device_palette(n: int) -> List:
    """Return ``n`` visually distinct colors for per-device assignments.

    ``tab20`` supplies 20 distinct colors; beyond that we sample a continuous
    map. This replaces the old fixed 10-colour list indexed with ``i % 10``,
    which gave two different devices the SAME colour once there were >10
    devices — making one device's *contiguous* block of layers look like
    several non-contiguous chunks in the partition graph.
    """
    if n <= 20:
        return list(plt.get_cmap("tab20").colors[:n])
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


def set_output_dir(output_dir: str) -> None:
    """Set up output directory for plots."""
    global _output_dir  # pylint: disable=global-statement
    _output_dir = output_dir


def get_output_dir() -> str:
    """Get the current output directory."""
    return _output_dir or "."


def draw_partitions(
    model: ModelSpec,
    net: Network,
    partitions: List[List[Dict]],
    node_weights: List[int],
    xadj: List[int],
    adjncy: List[int],
    edge_weights: List[int],
    attn_split_strategy: AttentionSplitStrategy,
    ffn_split_strategy: FFNSplitStrategy,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    policy_name: str = "METIS",
    full_assignment: Optional[List[int]] = None,
):
    """
    Visualize a partitioning policy's computation graph with device assignments.

    Shows:
    - Computation nodes (attention/FFN blocks) arranged by layer
    - Edges representing communication dependencies
    - Device assignments with color coding
    - Node weights (computation cost) as node sizes

    Args:
        policy_name: Display name of the partitioning policy (used in the title).
        full_assignment: Optional list mapping every node id (including virtual
            all-reduce nodes) to its device. When omitted, the device of any
            node not present in ``partitions`` (e.g. virtual nodes) is inferred
            from a majority vote over its neighbours so cross-edge colouring
            is correct.
    """

    # Create NetworkX graph
    graph = nx.DiGraph()

    # Calculate layout parameters
    k = len(net.servers)
    n_layers = model.num_layers
    max_attn_splits = model.get_max_attn_splits(attn_split_strategy)
    max_ffn_splits = model.get_max_ffn_splits(ffn_split_strategy, k)

    # Layer layout includes two virtual all-reduce nodes per layer:
    #   [attn_splits | ffn_splits | attn_AR | ffn_AR]
    nodes_per_layer = max_attn_splits + max_ffn_splits + 2
    ATTN_VIRT_OFFSET = max_attn_splits + max_ffn_splits  # ffn_AR is at +1

    # Build node to device mapping
    node_to_device: Dict[int, int] = {}
    for device_id, partition_nodes in enumerate(partitions):
        for node in partition_nodes:
            node_to_device[node["node_id"]] = device_id

    # If the caller supplied a full assignment (e.g. METIS, which assigns
    # virtual nodes too), use it for any node not already mapped.
    if full_assignment is not None:
        for nid, dev in enumerate(full_assignment):
            node_to_device.setdefault(nid, int(dev))

    # Any remaining missing nodes (virtual AR nodes from policies that don't
    # produce an explicit assignment) get the majority device of their
    # neighbours. This keeps virtual-AR edges from spuriously appearing as
    # cross-device when in fact every neighbour lives on the same device.
    from collections import Counter

    for nid in range(len(node_weights)):
        if nid in node_to_device:
            continue
        votes: List[int] = []
        for adj_idx in range(xadj[nid], xadj[nid + 1]):
            nbr = adjncy[adj_idx]
            if nbr in node_to_device:
                votes.append(node_to_device[nbr])
        node_to_device[nid] = (
            Counter(votes).most_common(1)[0][0] if votes else 0
        )

    # Add nodes to graph
    node_positions = {}
    node_colors = []
    node_sizes = []
    node_labels = {}

    # Distinct color per device (no wrap — see device_palette).
    device_colors = device_palette(len(net.servers))

    for node_id, node_weight in enumerate(node_weights):
        layer = node_id // nodes_per_layer
        within_layer = node_id % nodes_per_layer

        if within_layer < max_attn_splits:
            block_type = "attn"
            split_id = within_layer
            y_offset = split_id * 0.3
            x_base = layer * 2.0
        elif within_layer < ATTN_VIRT_OFFSET:
            block_type = "ffn"
            split_id = within_layer - max_attn_splits
            y_offset = split_id * 0.3
            x_base = layer * 2.0 + 0.8
        elif within_layer == ATTN_VIRT_OFFSET:
            # Virtual attn_AR node: sit just above the attention column
            block_type = "attn_AR"
            split_id = -1
            y_offset = -0.4
            x_base = layer * 2.0
        else:
            # Virtual ffn_AR node: sit just above the FFN column
            block_type = "ffn_AR"
            split_id = -1
            y_offset = -0.4
            x_base = layer * 2.0 + 0.8

        # Position nodes
        x = x_base
        y = y_offset
        node_positions[node_id] = (x, y)

        # Node properties
        device_id = node_to_device.get(node_id, 0)
        LOGGER.debug(
            "Node %s: %s_%s, Layer %s, Device %s",
            node_id,
            block_type,
            split_id,
            layer,
            device_id,
        )
        node_colors.append(device_colors[device_id])

        # Scale node size by computation weight
        base_size = 300
        weight_scale = (
            np.log(max(1, node_weight)) / np.log(max(node_weights))
            if max(node_weights) > 0
            else 1
        )
        node_sizes.append(base_size + weight_scale * 500)

        # Node labels
        node_labels[node_id] = (
            f"{block_type}_{split_id}"  # f"L{layer}_{block_type}_{split_id}"
        )

        # Add to graph
        graph.add_node(
            node_id,
            layer=layer,
            block_type=block_type,
            split_id=split_id,
            device_id=device_id,
            weight=node_weights[node_id],
        )

    # Add edges
    edge_colors = []
    edge_widths = []

    for node_id in range(len(node_weights)):
        start_idx = xadj[node_id]
        end_idx = xadj[node_id + 1]

        for adj_idx in range(start_idx, end_idx):
            neighbor_id = adjncy[adj_idx]
            edge_weight = edge_weights[adj_idx]

            graph.add_edge(node_id, neighbor_id, weight=edge_weight)

            # Color edges based on whether they cross devices
            source_device = node_to_device.get(node_id, 0)
            target_device = node_to_device.get(neighbor_id, 0)

            if source_device == target_device:
                edge_colors.append("lightgray")
                edge_widths.append(1)
            else:
                edge_colors.append("red")
                edge_widths.append(3)

    ax1: Axes
    ax2: Axes
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1], "wspace": 0}
    )

    # Main graph plot
    nx.draw_networkx_nodes(
        graph,
        node_positions,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax1,
    )

    # Draw edges separately by type
    intra_device_edges = [
        (u, v)
        for i, (u, v) in enumerate(graph.edges())
        if edge_colors[i] == "lightgray"
    ]
    cross_device_edges = [
        (u, v) for i, (u, v) in enumerate(graph.edges()) if edge_colors[i] == "red"
    ]

    if intra_device_edges:
        nx.draw_networkx_edges(
            graph,
            node_positions,
            edgelist=intra_device_edges,
            edge_color="lightgray",
            width=1,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax1,
        )

    if cross_device_edges:
        nx.draw_networkx_edges(
            graph,
            node_positions,
            edgelist=cross_device_edges,
            edge_color="red",
            width=3,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax1,
        )

    nx.draw_networkx_labels(graph, node_positions, node_labels, font_size=8, ax=ax1)

    ax1.set_title(
        f"{policy_name} Computation Graph\n"
        f"{n_layers} layers, {len(net.servers)} devices",
        fontsize=14,
        fontweight="bold",
    )
    ax1.axis("off")

    # Add layer labels
    for layer in range(n_layers):
        x = layer * 2.0 + 0.4
        y = -0.2
        ax1.text(x, y, f"Layer {layer}", ha="center", fontsize=10, fontweight="bold")

    # Device assignment legend
    ax2.set_title("Device Assignments", fontsize=12, fontweight="bold", loc="left")

    # Create legend for devices
    legend_elements = []
    for device_id, device in enumerate(net.servers):
        # Count nodes assigned to this device
        assigned_nodes = sum(
            1 for _, dev_id in node_to_device.items() if dev_id == device_id
        )

        legend_elements.append(
            patches.Rectangle(
                (0, 0),
                2,
                1,
                facecolor=device_colors[device_id],
                label=f"{device.name}\n({assigned_nodes} nodes)",
            )
        )

    # Wrap the device legend into columns of at most 6 entries so it doesn't
    # grow into a tall single column that overlaps the statistics text below.
    ncol = max(1, -(-len(legend_elements) // 6))  # ceil(n / 6)
    # Anchor at the same left x as the statistics text below (-0.1) and pin to
    # the top, so the legend sits directly above the stats block rather than
    # floating to the right of it.
    ax2.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(-0.1, 1.0),
        fontsize=10,
        ncol=ncol,
        handlelength=1.2,
        columnspacing=1.0,
        labelspacing=0.4,
        borderaxespad=0.0,
    )
    ax2.axis("off")

    # Add communication statistics
    cross_device_edges = sum(1 for color in edge_colors if color == "red")
    intra_device_edges = sum(1 for color in edge_colors if color == "lightgray")

    stats_text = f"""Graph Statistics:
Total Nodes: {len(node_weights)}
Total Edges: {len(edge_colors)}
Cross-device Edges: {cross_device_edges}
Intra-device Edges: {intra_device_edges}
Communication Ratio: {cross_device_edges/(cross_device_edges+intra_device_edges):.2%}

Legend:
● Gray edges: Same device
● Red edges: Cross device
● Node size ∝ Computation cost"""

    ax2.text(
        -0.1,
        -0.01,
        stats_text,
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="bottom",
    )
    if not save_path:
        save_path = f"{get_output_dir()}/partition_graph.pdf"

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Graph saved to %s", save_path)
    _emit_figure(fig)

    return fig


def draw_device_computation_breakdown(
    partitions: List[List[Dict]],
    servers: List[Device],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Create a bar chart showing computation breakdown per device.
    """

    device_names = [server.name for server in servers]
    attention_counts = []
    ffn_counts = []
    total_weights = []

    for partition_nodes in partitions:
        attn_count = sum(
            1 for node in partition_nodes if node.get("block_type") == "attn"
        )
        ffn_count = sum(
            1 for node in partition_nodes if node.get("block_type") == "ffn"
        )
        total_weight = sum(node.get("compute_weight", 0) for node in partition_nodes)

        attention_counts.append(attn_count)
        ffn_counts.append(ffn_count)
        total_weights.append(total_weight)

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Block count chart
    x = np.arange(len(device_names))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        attention_counts,
        width,
        label="Attention Blocks",
        color="skyblue",
    )
    bars2 = ax1.bar(
        x + width / 2, ffn_counts, width, label="FFN Blocks", color="lightcoral"
    )

    ax1.set_xlabel("Device")
    ax1.set_ylabel("Number of Blocks")
    ax1.set_title("Computation Block Assignment per Device")
    ax1.set_xticks(x)
    ax1.set_xticklabels(device_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar1 in bars1:
        height = bar1.get_height()
        if height > 0:
            ax1.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

    for bar2 in bars2:
        height = bar2.get_height()
        if height > 0:
            ax1.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

    # Computation weight chart
    bars3 = ax2.bar(device_names, total_weights, color="lightgreen")
    ax2.set_xlabel("Device")
    ax2.set_ylabel("Total Computation Weight")
    ax2.set_title("Computation Load Distribution")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar3 in bars3:
        height = bar3.get_height()
        if height > 0:
            ax2.text(
                bar3.get_x() + bar3.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if not save_path:
        save_path = f"{get_output_dir()}/device_computation_breakdown.pdf"
    if SAVE_FIGURES:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Chart saved to %s", save_path)
    _emit_figure(fig)

    return fig


def plot_resource_utilization(
    resources: List[str],
    resource_units: List[str],
    totals: List[float],
    servers: List[Device],
    constraining_resource: List[str],
    assignments: List[float],
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
):
    """Plot resource utilization for all devices per resource."""
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(resources))

    dev_colors = device_palette(len(assignments))

    # Prepare stacked bar data
    for i, (unit, total) in enumerate(zip(resource_units, totals)):  # per resource
        server_allocs = np.array(assignments) * np.array([total] * len(assignments))
        print(f"Resource: {resources[i]}, Total: {total}, Allocations: {server_allocs}")
        sum_pcts = 0
        for server_id, server_pct in enumerate(assignments):  # per server
            server_pct *= 100
            ax.bar(
                x,
                server_pct,
                bottom=sum_pcts,
                color=dev_colors[server_id % len(dev_colors)],
                label=servers[server_id].name if i == 0 else "",  # only label once
            )

            # Annotate non-empty segments only (skip the 0% devices so their
            # labels don't pile up on each other), placing the value just ABOVE
            # the segment rather than centred on it / on the 50% line.
            if server_pct >= 1.0:
                ax.text(
                    i,
                    sum_pcts + server_pct + 1.5,
                    f"{server_allocs[server_id]:.2f} {unit}",
                    color=(
                        "red"
                        if resources[i] == constraining_resource[server_id]
                        else "black"
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            sum_pcts += server_pct

    # Aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r}\n({t:.2f} total)" for r, t in zip(resources, totals)], fontsize=10
    )
    ax.set_ylabel("Utilization (%)", fontsize=11)
    ax.set_ylim(0, 112)  # headroom so the value labels above each bar aren't clipped
    ax.axhline(50, linestyle="--", color="black", linewidth=1)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # outside, to the right of the axes
        ncol=1,
        fontsize=7,
        title="device",
    )
    plt.tight_layout()

    if not save_path:
        save_path = f"{get_output_dir()}/device_computation_breakdown.pdf"
    if SAVE_FIGURES:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Chart saved to %s", save_path)
    _emit_figure(fig)

    return fig
