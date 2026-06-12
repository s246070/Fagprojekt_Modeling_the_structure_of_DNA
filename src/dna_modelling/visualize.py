# os.environ["MPLBACKEND"] = "TkAgg"

# matplotlib.use("TkAgg", force=True)

import numpy as np
import scipy.sparse as sp
# import igraph as ig
# from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import umap

def plot_bipartite_graph(adata, output_path, max_cells_plot=120, max_features_plot=120):
    """
    Create and save a bipartite graph visualization from an AnnData object.

    Args:
        adata: AnnData object with a cells x features matrix in adata.X.
        output_path: Path where the figure will be saved.
        max_cells_plot: Maximum number of cell nodes to include in the plot.
        max_features_plot: Maximum number of feature nodes to include in the plot.

    Attributes:
         graph: The constructed igraph.Graph object representing the bipartite graph.

    Returns:
        The constructed igraph.Graph object.
    """
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)

    rows, cols, _ = sp.find(X_bin)

    n_cells, n_features = X_bin.shape
    feature_offset = n_cells

    edges = list(zip(rows.tolist(), (cols + feature_offset).tolist()))
    graph = ig.Graph(n=n_cells + n_features, edges=edges, directed=False)

    graph.vs["type"] = [False] * n_cells + [True] * n_features

    _, _, orig_vals = sp.find(X)
    graph.es["weight"] = orig_vals.tolist()

    cell_nodes = list(range(min(n_cells, max_cells_plot)))
    feature_nodes = [
        feature_offset + i for i in range(min(n_features, max_features_plot))
    ]
    selected_nodes = set(cell_nodes + feature_nodes)

    sub_edge_ids = [
        edge.index
        for edge in graph.es
        if edge.tuple[0] in selected_nodes and edge.tuple[1] in selected_nodes
    ]

    graph_plot = graph.subgraph_edges(sub_edge_ids, delete_vertices=True)

    node_colors = ["#1f77b4" if not node_type else "#ff7f0e" for node_type in graph_plot.vs["type"]]
    layout = graph_plot.layout_bipartite(types=graph_plot.vs["type"])

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.plot(
        graph_plot,
        target=ax,
        layout=layout,
        vertex_size=6,
        vertex_color=node_colors,
        edge_width=0.4,
        edge_color="gray",
        margin=40,
        bbox=(900, 700),
    )

    ax.set_title(
        f"Bipartite graph subset: {sum(~np.array(graph_plot.vs['type']))} cells and "
        f"{sum(np.array(graph_plot.vs['type']))} features"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return graph

def plot_latent_embeddings(model, ls_dim, epoch):
    """
    Create and save UMAP visualizations of the latent embeddings for cells and peaks.

    Args:
        model: The trained model containing the cell and peak embeddings.
        ls_dim: The dimension of the latent space.
        epoch: The current training epoch.

    Returns:
        None
    """
    Path("plots").mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
    )

    cell_embeddings = model.embed_cells.detach().cpu().numpy()
    peak_embeddings = model.embed_features.detach().cpu().numpy()
    stacked_embeddings = np.vstack([cell_embeddings, peak_embeddings])
    stacked_embeddings_2d = reducer.fit_transform(stacked_embeddings)

    n_cells = cell_embeddings.shape[0]
    cell_embeddings_2d = stacked_embeddings_2d[:n_cells]
    peak_embeddings_2d = stacked_embeddings_2d[n_cells:]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        cell_embeddings_2d[:, 0],
        cell_embeddings_2d[:, 1],
        s=5,
        alpha=0.7,
        color="blue",
        label="Cells",
    )
    plt.scatter(
        peak_embeddings_2d[:, 0],
        peak_embeddings_2d[:, 1],
        s=5,
        alpha=0.7,
        color="red",
        label="Peaks",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Cell & Peak latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        cell_embeddings_2d[:, 0],
        cell_embeddings_2d[:, 1],
        s=5,
        alpha=0.7,
        color="blue",
        label="Cells",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Cell latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_cells_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        peak_embeddings_2d[:, 0],
        peak_embeddings_2d[:, 1],
        s=5,
        alpha=0.7,
        color="red",
        label="Peaks",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Peak latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_peaks_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

def plot_embeddings(model, ls_dim, epoch, interval_steps, losses_per_interval):
    cell_embeddings = model.embed_cells.detach().cpu().numpy()
    peak_embeddings = model.embed_features.detach().cpu().numpy()
    plt.figure(figsize=(7, 6))
    plt.scatter(
        cell_embeddings[:, 0],
        cell_embeddings[:, 1],
        s=5,
        alpha=0.7,
        color="blue",
        label="Cells",
    )
    plt.scatter(
        peak_embeddings[:, 0],
        peak_embeddings[:, 1],
        s=5,
        alpha=0.7,
        color="red",
        label="Peaks",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Cell & Peak latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        cell_embeddings[:, 0],
        cell_embeddings[:, 1],
        s=5,
        alpha=0.7,
        color="blue",
        label="Cells",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Cell latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_cells_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        peak_embeddings[:, 0],
        peak_embeddings[:, 1],
        s=5,
        alpha=0.7,
        color="red",
        label="Peaks",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(f"Peak latent space (ls_dim={ls_dim})")
    plt.legend()
    plt.savefig(f"plots/latent_space_peaks_ls{ls_dim}_{epoch}.png", dpi=300)
    plt.close()

def plot_loss_curve(ls_dim, interval_steps, losses_per_interval):
    """
    Create and save a plot of the training loss over time.

    Args:
        ls_dim: The dimension of the latent space.
        interval_steps: A list of epoch numbers at specified intervals.
        losses_per_interval: A list of loss values at specified intervals.

    Returns:
        None
    """
    plt.figure(figsize=(7, 6))
    plt.plot(interval_steps, losses_per_interval)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (ls_dim={ls_dim})")
    plt.grid(True)
    plt.savefig(f"plots/loss_curve_ls{ls_dim}.png", dpi=300)
    plt.close()