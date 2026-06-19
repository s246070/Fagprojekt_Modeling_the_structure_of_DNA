from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# -----------------------------
# Settings
# -----------------------------
name = "pca_subset_embeddings"
embedding_path = f"models/{name}.npy"

# Use subset labels if the PCA file only contains the 10,000-cell subset
cell_type_path = "src/benchmarking/cell_types_subset_1.txt"
# cell_type_path = "src/benchmarking/cell_types.txt"

save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load PCA embeddings
# -----------------------------
data = np.load(embedding_path)

if data.ndim == 1:
    data = data[:, None]

print(f"Loaded {data.shape[0]} cells")
print(f"PCA dimension: {data.shape[1]}")


# -----------------------------
# Load cell types
# -----------------------------
with open(cell_type_path, "r") as f:
    cell_types = np.array([line.strip() for line in f])

assert len(cell_types) == data.shape[0], (
    f"Number of labels ({len(cell_types)}) does not match "
    f"number of PCA embeddings ({data.shape[0]})"
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
        for label in unique_cell_types
    ]


# -----------------------------
# 2D PCA plot
# -----------------------------
def plot_pca_2d():
    if data.shape[1] < 2:
        print(f"Skipping 2D plot because PCA dimension is {data.shape[1]}")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for label in unique_cell_types:
        mask = cell_types == label

        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            s=2,
            color=label_to_color[label],
            alpha=0.75,
            rasterized=True,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

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

    plt.savefig(
        save_dir / f"{name}_2d.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()


# -----------------------------
# 3D PCA plot
# -----------------------------
def plot_pca_3d():
    if data.shape[1] < 3:
        print(f"Skipping 3D plot because PCA dimension is {data.shape[1]}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in unique_cell_types:
        mask = cell_types == label

        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            data[mask, 2],
            s=2,
            color=label_to_color[label],
            alpha=0.8,
            rasterized=True,
        )

    ax.set_xlabel("PC1", labelpad=0)
    ax.set_ylabel("PC2", labelpad=0)
    ax.set_zlabel("PC3", labelpad=0)

    ax.grid(True)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=20, azim=130)

    # Uncomment if you want legend on the 3D plot
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

    plt.savefig(
        save_dir / f"{name}_3d.png",
        dpi=600,
    )

    plt.show()


# -----------------------------
# Run plots
# -----------------------------
plot_pca_2d()
plot_pca_3d()