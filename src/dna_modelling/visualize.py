# import all necessary libraries to open .h5ad files and perform data analysis
import os
os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib
matplotlib.use("TkAgg", force=True)

import numpy as np
import scipy.sparse as sp
import igraph as ig
from scipy.sparse import csr_matrix
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import umap

#Change what graph you want printed: 1 is for the bipartite graph, 2 is for the latent space visualization
graph_to_print = 2


if graph_to_print == 1:
    # load the .h5ad file into an AnnData object
    adata = sc.read_h5ad('C:\\Users\\Peter\\OneDrive\\Documents\\GitHub\\Fagprojekt_Modeling_the_structure_of_DNA\\data\\atac_pbmc_5k_preprocessed.h5ad')
    # check the structure of the data
    print(adata)

    test_data = [[0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1]]

    # Implement bipartite graph-based clustering using the Leiden algorithm
    # First, we need to create a bipartite graph from the AnnData object
    # We will use the 'X' matrix (which contains the gene expression data) to create the graph

    # Create a bipartite graph using the gene expression data
    matrix_thing = csr_matrix(test_data)
    print(matrix_thing)

    # Build a bipartite graph directly from adata.X (cells x features)

    # 1) Use the existing incidence matrix (non-zero entries are edges)
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    # Optional: binarize if you only want connectivity, not weights
    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)

    # 2) Extract edges from the sparse matrix
    rows, cols, vals = sp.find(X_bin)  # rows=cells, cols=features

    n_cells, n_features = X_bin.shape
    feature_offset = n_cells

    # 3) Create bipartite graph in igraph
    edges = list(zip(rows.tolist(), (cols + feature_offset).tolist()))
    g = ig.Graph(n=n_cells + n_features, edges=edges, directed=False)

    # Mark bipartite node types: False=cell, True=feature
    g.vs["type"] = [False] * n_cells + [True] * n_features

    # Keep original matrix values as edge weights (if needed)
    orig_rows, orig_cols, orig_vals = sp.find(X)
    g.es["weight"] = orig_vals.tolist()

    print(f"Bipartite graph created with {g.vcount()} nodes and {g.ecount()} edges")
    print(f"Cells: {n_cells}, Features: {n_features}")

    # Plot a readable subset of the bipartite graph
    # Full graph can be very dense; limit nodes for visualization
    max_cells_plot = 120
    max_features_plot = 120

    cell_nodes = list(range(min(n_cells, max_cells_plot)))
    feature_nodes = [feature_offset + i for i in range(min(n_features, max_features_plot))]
    selected_nodes = set(cell_nodes + feature_nodes)

    # Keep edges where both endpoints are in the selected node set
    sub_edge_ids = [
        e.index for e in g.es
        if e.tuple[0] in selected_nodes and e.tuple[1] in selected_nodes
    ]

    g_plot = g.subgraph_edges(sub_edge_ids, delete_vertices=True)

    # Color by partition: cells vs features
    node_colors = ["#1f77b4" if not t else "#ff7f0e" for t in g_plot.vs["type"]]

    layout = g_plot.layout_bipartite(types=g_plot.vs["type"])

    fig, ax = plt.subplots(figsize=(11, 8))
    ig.plot(
        g_plot,
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
        f"Bipartite graph subset: {sum(~np.array(g_plot.vs['type']))} cells and "
        f"{sum(np.array(g_plot.vs['type']))} features"
    )

    output_path = Path("C:\\Users\\Peter\\OneDrive\\Documents\\GitHub\\Fagprojekt_Modeling_the_structure_of_DNA\\data\\bipartite_graph_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved bipartite graph figure to: {output_path}")

    plt.show()
elif graph_to_print == 2:
    from dna_modelling.latent_space_visualization import load_edges, load_nodes, plot_latent_space

    #Change paths to where your files are located
    nodes_path = Path("C:\\Users\\Peter\\OneDrive\\Documents\\GitHub\\Fagprojekt_Modeling_the_structure_of_DNA\\data\\latent_space_nodes.csv")
    edges_path = Path("C:\\Users\\Peter\\OneDrive\\Documents\\GitHub\\Fagprojekt_Modeling_the_structure_of_DNA\\data\\latent_space_edges.csv")
    output_path = Path("C:\\Users\\Peter\\OneDrive\\Documents\\GitHub\\Fagprojekt_Modeling_the_structure_of_DNA\\data\\latent_space_plot_from_notebook.png")

    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    plot_latent_space(nodes=nodes, edges=edges, output_path=output_path)

    print(f"Saved latent-space figure to: {output_path}")

def plot_latent_embeddings(model, ls_dim, epoch):
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

def plot_loss_curve(ls_dim, interval_steps, losses_per_interval):
    plt.figure(figsize=(7, 6))
    plt.plot(interval_steps, losses_per_interval)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (ls_dim={ls_dim})")
    plt.grid(True)
    plt.savefig(f"plots/loss_curve_ls{ls_dim}.png", dpi=300)
    plt.close()