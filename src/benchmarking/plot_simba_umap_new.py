from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scanpy as sc
import umap

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# -----------------------------
# Settings
# -----------------------------
name = "simba_subset_dim16"
adata_path = "models/adata_cells_ldm16_simba.h5ad"

save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load SIMBA cell embeddings
# -----------------------------
adata = sc.read_h5ad(adata_path)

latent = adata.X
if hasattr(latent, "toarray"):
    latent = latent.toarray()

latent = np.asarray(latent)

labels = adata.obs["cell_type"].to_numpy()
unique_labels = np.unique(labels)

assert len(labels) == latent.shape[0], (
    f"Number of labels ({len(labels)}) does not match "
    f"number of latent embeddings ({latent.shape[0]})"
)

print(f"Loaded {latent.shape[0]} cells")
print(f"Latent dimension: {latent.shape[1]}")


# -----------------------------
# Colors
# -----------------------------
cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))

label_to_color = {
    label: cmap(i)
    for i, label in enumerate(unique_labels)
}


# -----------------------------
# Helpers
# -----------------------------
def remove_axis_numbers(ax, is_3d=False):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if is_3d:
        ax.set_zticklabels([])


def make_legend_handles():
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=label_to_color[label],
            markeredgecolor=label_to_color[label],
            markersize=6,
            label=label,
        )
        for label in unique_labels
    ]


def compute_umap(
    n_components,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
    )

    return reducer.fit_transform(latent)


# -----------------------------
# 2D UMAP plot
# -----------------------------
def plot_umap_2d(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    embedding = compute_umap(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=2,
            color=label_to_color[label],
            alpha=0.75,
            rasterized=True,
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    remove_axis_numbers(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=make_legend_handles(),
        title="Cell Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    filename = f"{name}_umap2d_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.show()


# -----------------------------
# 3D UMAP plot
# -----------------------------
def plot_umap_3d(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    embedding = compute_umap(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=2,
            color=label_to_color[label],
            alpha=0.8,
            rasterized=True,
        )

    ax.set_xlabel("UMAP1", labelpad=0)
    ax.set_ylabel("UMAP2", labelpad=0)
    ax.set_zlabel("UMAP3", labelpad=0)

    ax.grid(True)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=20, azim=130)

    # Uncomment if you want legend on 3D plot
    # ax.legend(
    #     handles=make_legend_handles(),
    #     title="Cell Types",
    #     bbox_to_anchor=(1.1, 1),
    #     loc="upper left",
    #     frameon=False,
    # )

    plt.subplots_adjust(
        left=-0.1,
        right=1.05,
        bottom=-0.1,
        top=1.1,
    )

    filename = f"{name}_umap3d_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600)
    plt.show()


# -----------------------------
# Run plots
# -----------------------------

plot_umap_2d(n_neighbors=15, min_dist=0.5)
plot_umap_3d(n_neighbors=15, min_dist=0.5)