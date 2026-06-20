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

cell_adata_path = "results/simba/simba_subset_dim{16}/adata_cells_ldm16_simba.h5ad"
feature_adata_path = "results/simba/simba_subset_dim{16}/adata_peaks_ldm16_simba.h5ad"

save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load SIMBA cell embeddings
# -----------------------------
cell_adata = sc.read_h5ad(cell_adata_path)
feature_adata = sc.read_h5ad(feature_adata_path)

cell_embeddings = cell_adata.X
feature_embeddings = feature_adata.X

if hasattr(cell_embeddings, "toarray"):
    cell_embeddings = cell_embeddings.toarray()

if hasattr(feature_embeddings, "toarray"):
    feature_embeddings = feature_embeddings.toarray()

cell_embeddings = np.asarray(cell_embeddings)
feature_embeddings = np.asarray(feature_embeddings)

labels = cell_adata.obs["cell_type"].to_numpy()
unique_labels = np.unique(labels)

assert len(labels) == cell_embeddings.shape[0], (
    f"Number of labels ({len(labels)}) does not match "
    f"number of cell embeddings ({cell_embeddings.shape[0]})"
)

assert cell_embeddings.shape[1] == feature_embeddings.shape[1], (
    f"Cell embedding dim ({cell_embeddings.shape[1]}) does not match "
    f"feature embedding dim ({feature_embeddings.shape[1]})"
)

print(f"Loaded {cell_embeddings.shape[0]} cells")
print(f"Loaded {feature_embeddings.shape[0]} features")
print(f"Latent dimension: {cell_embeddings.shape[1]}")


# -----------------------------
# Colors
# -----------------------------
cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))

label_to_color = {
    label: cmap(i)
    for i, label in enumerate(unique_labels)
}

feature_color = "black"


# -----------------------------
# Helpers
# -----------------------------
def remove_axis_numbers(ax, is_3d=False):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if is_3d:
        ax.set_zticklabels([])


def make_legend_handles(include_features=True):
    handles = []

    if include_features:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=feature_color,
                markeredgecolor=feature_color,
                markersize=5,
                label="Features",
            )
        )

    for label in unique_labels:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=label_to_color[label],
                markeredgecolor=label_to_color[label],
                markersize=6,
                label=label,
            )
        )

    return handles


def compute_joint_umap(
    n_components,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    combined_embeddings = np.vstack([
        cell_embeddings,
        feature_embeddings,
    ])

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
    )

    combined_umap = reducer.fit_transform(combined_embeddings)

    cell_umap = combined_umap[:cell_embeddings.shape[0]]
    feature_umap = combined_umap[cell_embeddings.shape[0]:]

    return cell_umap, feature_umap


# -----------------------------
# 2D joint UMAP plot
# -----------------------------
def plot_umap_2d(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    cell_umap, feature_umap = compute_joint_umap(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        feature_umap[:, 0],
        feature_umap[:, 1],
        s=0.5,
        c=feature_color,
        alpha=0.1,
        rasterized=True,
    )

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            cell_umap[mask, 0],
            cell_umap[mask, 1],
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
        handles=make_legend_handles(include_features=True),
        title="Cell Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    filename = f"{name}_joint_umap2d_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.show()


# -----------------------------
# 3D joint UMAP plot
# -----------------------------
def plot_umap_3d(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    cell_umap, feature_umap = compute_joint_umap(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        feature_umap[:, 0],
        feature_umap[:, 1],
        feature_umap[:, 2],
        s=0.5,
        c=feature_color,
        alpha=0.15,
        rasterized=True,
    )

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            cell_umap[mask, 0],
            cell_umap[mask, 1],
            cell_umap[mask, 2],
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

    ax.set_xlim(3, 20)
    ax.set_ylim(-10, 8)
    ax.set_zlim(-5, 12)

    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=20, azim=130)

    plt.subplots_adjust(
        left=-0.1,
        right=1.05,
        bottom=-0.1,
        top=1.1,
    )

    filename = f"{name}_joint_umap3d_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600)
    plt.show()


# -----------------------------
# Run plots
# -----------------------------
plot_umap_3d(n_neighbors=15, min_dist=0.5)