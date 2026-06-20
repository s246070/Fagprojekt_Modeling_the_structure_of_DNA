from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import umap

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# -----------------------------
# Settings
# -----------------------------
name = "ldm_ls16_epoch1000_blocks10_index1"
model_path = f"models/ldm_ls16_epoch1000_blocks10_index1.pth"
cell_type_path = "src/benchmarking/cell_types_subset_1.txt"

save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load LDM embeddings
# -----------------------------
model = torch.load(model_path, map_location="cpu")

cell_embeddings = model["embed_cells"].detach().cpu().numpy()
feature_embeddings = model["embed_features"].detach().cpu().numpy()

print(f"Loaded {cell_embeddings.shape[0]} cells")
print(f"Loaded {feature_embeddings.shape[0]} features")
print(f"Latent dimension: {cell_embeddings.shape[1]}")


# -----------------------------
# Load cell types
# -----------------------------
with open(cell_type_path, "r") as f:
    cell_types = np.array([line.strip() for line in f])

assert len(cell_types) == cell_embeddings.shape[0], (
    f"Number of labels ({len(cell_types)}) does not match "
    f"number of cell embeddings ({cell_embeddings.shape[0]})"
)

unique_cell_types = np.unique(cell_types)


# -----------------------------
# Colors
# -----------------------------
cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cell_types))

label_to_color = {
    label: cmap(i)
    for i, label in enumerate(unique_cell_types)
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

    for label in unique_cell_types:
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

    # Features behind cells
    ax.scatter(
        feature_umap[:, 0],
        feature_umap[:, 1],
        s=0.5,
        c=feature_color,
        alpha=0.07,
        rasterized=True,
    )

    # Cells by type
    for label in unique_cell_types:
        mask = cell_types == label

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

    # Features behind cells
    ax.scatter(
        feature_umap[:, 0],
        feature_umap[:, 1],
        feature_umap[:, 2],
        s=0.5,
        c=feature_color,
        alpha=0.15,
        rasterized=True,
    )

    # Cells by type
    for label in unique_cell_types:
        mask = cell_types == label

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

    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=20, azim=130)

    # Uncomment if you want the legend on 3D plots
    # ax.legend(
    #     handles=make_legend_handles(include_features=True),
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

    filename = f"{name}_joint_umap3d_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600)
    plt.show()


# -----------------------------
# Run plots
# -----------------------------
for n_neighbors in [5, 15, 50]:
    for min_dist in [0.1, 0.5, 0.9]:
        plot_umap_2d(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

        plot_umap_3d(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
# ------------------------------
# Run with specific settings
# ------------------------------
# plot_umap_2d(n_neighbors=15, min_dist=0.1)
# plot_umap_3d(n_neighbors=15, min_dist=0.1)