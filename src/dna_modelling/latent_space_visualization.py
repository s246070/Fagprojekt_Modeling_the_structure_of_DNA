"""Create a latent-space visualization from node and edge CSV files."""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class Node:
    """
    Store node attributes for plotting.

    Attributes:
        node_id: Unique node identifier.
        label: Group label.
        x: X coordinate in latent space.
        y: Y coordinate in latent space.
    """
    node_id: str
    label: str
    x: float
    y: float

@dataclass
class Edge:
    """
    Store edge attributes for plotting.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        weight: Edge weight.
    """
    source: str
    target: str
    weight: float

def load_nodes(nodes_path: Path) -> dict[str, Node]:
    """
    Load node data from CSV.

    Args:
        nodes_path: Path to node CSV file.

    Returns:
        Mapping from node ID to Node.
    """
    nodes: dict[str, Node] = {}
    with nodes_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            node = Node(
                node_id=row["node_id"],
                label=row["label"],
                x=float(row["latent_x_true"]),
                y=float(row["latent_y_true"]),
            )
            nodes[node.node_id] = node
    return nodes

def load_edges(edges_path: Path) -> list[Edge]:
    """
    Load edge data from CSV.

    Args:
        edges_path: Path to edge CSV file.

    Returns:
        List of edges.
    """
    edges: list[Edge] = []
    with edges_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            edges.append(
                Edge(
                    source=row["source"],
                    target=row["target"],
                    weight=float(row["weight"]),
                )
            )
    return edges

def plot_latent_space(nodes: dict[str, Node], edges: list[Edge], output_path: Path) -> None:
    """
    Create and save latent space plot.

    Args:
        nodes: Mapping of node IDs to nodes.
        edges: List of weighted edges.
        output_path: Path where the figure is written.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for edge in edges:
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
        if source_node is None or target_node is None:
            continue

        ax.plot(
            [source_node.x, target_node.x],
            [source_node.y, target_node.y],
            color="gray",
            alpha=0.35,
            linewidth=0.6 + 0.4 * edge.weight,
            zorder=1,
        )

    labels = sorted({node.label for node in nodes.values()})
    cmap = plt.get_cmap("tab10")

    for index, label in enumerate(labels):
        label_nodes = [node for node in nodes.values() if node.label == label]
        ax.scatter(
            [node.x for node in label_nodes],
            [node.y for node in label_nodes],
            label=label,
            color=cmap(index % 10),
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )

    for node in nodes.values():
        ax.text(node.x + 0.05, node.y + 0.05, node.node_id, fontsize=8)

    ax.set_title("Latent Space Graph")
    ax.set_xlabel("Latent dimension x")
    ax.set_ylabel("Latent dimension y")
    ax.grid(alpha=0.2)
    ax.legend(title="Label")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize latent-space nodes and edges from CSV files.")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/latent_space_nodes.csv"),
        help="Path to nodes CSV file.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/latent_space_edges.csv"),
        help="Path to edges CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/latent_space_plot.png"),
        help="Path to save output figure.",
    )
    return parser.parse_args()

def main() -> None:
    """
    Run CLI for latent-space visualization.
    """
    args = parse_args()
    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)
    plot_latent_space(nodes=nodes, edges=edges, output_path=args.output)
    print(f"Saved latent-space figure to: {args.output}")


if __name__ == "__main__":
    main()