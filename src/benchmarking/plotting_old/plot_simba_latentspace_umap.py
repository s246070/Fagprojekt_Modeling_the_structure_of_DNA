from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import umap

data = sc.read_h5ad("results/simba/simba_subset_dim{32}/adata_cells_ldm32_simba.h5ad")

latent = data.X

name = "simba_subset_dim32"

Path("plots").mkdir(parents=True, exist_ok=True)

labels = data.obs["cell_type"].to_numpy()
unique_labels = np.unique(labels)

color_map = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

colors = [label_to_color[label] for label in labels]


def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(latent)
    fig = plt.figure()

    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], np.zeros(len(u)), c=colors, cmap="viridis", s=10)
        ax.set_yticks([])
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c=colors, cmap="viridis", s=10)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=colors, cmap="viridis", s=10)
    plt.title(title, fontsize=18)
    plt.xlabel("UMAP 1", fontsize=14)
    plt.ylabel("UMAP 2", fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"plots/{name}_umap_{n_neighbors}_{min_dist}.png", dpi=300)
    plt.close(fig)

for n_neighbors in [5, 15, 50]:
    for min_dist in [0.1, 0.5, 0.9]:
        draw_umap(n_neighbors=n_neighbors, min_dist=min_dist, title=f"{name}, n_neighbors={n_neighbors}, min_dist={min_dist}")