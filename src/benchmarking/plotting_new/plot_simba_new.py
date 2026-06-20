from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# -----------------------------
# Settings
# -----------------------------
name = "simba_subset_dim3"

cell_adata_path = "results/simba/simba_subset_dim{3}/adata_cells_ldm3_simba.h5ad"
feature_adata_path = "results/simba/simba_subset_dim{3}/adata_peaks_ldm3_simba.h5ad"

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



# -----------------------------
# 2D joint SIMBA plot
# -----------------------------
def plot_simba_2d():

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        feature_embeddings[:, 0],
        feature_embeddings[:, 1],
        s=0.5,
        c=feature_color,
        alpha=0.1,
        rasterized=True,
    )

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            cell_embeddings[mask, 0],
            cell_embeddings[mask, 1],
            s=2,
            color=label_to_color[label],
            alpha=0.75,
            rasterized=True,
        )

    ax.set_xlabel("SIMBA1")
    ax.set_ylabel("SIMBA2")

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

    filename = f"{name}_joint_latent_2d.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.show()


# -----------------------------
# 3D joint SIMBA plot
# -----------------------------
def plot_simba_3d():

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        feature_embeddings[:, 0],
        feature_embeddings[:, 1],
        feature_embeddings[:, 2],
        s=0.5,
        c=feature_color,
        alpha=0.15,
        rasterized=True,
    )

    for label in unique_labels:
        mask = labels == label

        ax.scatter(
            cell_embeddings[mask, 0],
            cell_embeddings[mask, 1],
            cell_embeddings[mask, 2],
            s=2,
            color=label_to_color[label],
            alpha=0.8,
            rasterized=True,
        )

    ax.set_xlabel("SIMBA1", labelpad=0)
    ax.set_ylabel("SIMBA2", labelpad=0)
    ax.set_zlabel("SIMBA3", labelpad=0)

    ax.grid(True)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    ax.set_xlim(-2.5, 2)
    ax.set_ylim(-4, 0.5)
    ax.set_zlim(-3, 1.75)

    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=30, azim=130)

    plt.subplots_adjust(
        left=-0.1,
        right=1.1,
        bottom=-0,
        top=1.1,
    )

    filename = f"{name}_joint_simba3d.png"
    plt.savefig(save_dir / filename, dpi=600)
    plt.show()


# -----------------------------
# Run plots
# -----------------------------
plot_simba_3d()