from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import umap


# -----------------------------
# Settings
# -----------------------------
name = "peakvi_latent_16d"
model_path = f"models/peakvi_latent_16d_subset_10k.pth"
cell_type_path = "src/benchmarking/cell_types.txt"

save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load PeakVI latent embeddings
# -----------------------------
data = torch.load(model_path, map_location="cpu")

latent = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)

if latent.ndim == 1:
    latent = latent[:, None]
elif latent.ndim > 2:
    latent = latent.reshape(latent.shape[0], -1)

print(f"Loaded {latent.shape[0]} cells")
print(f"Latent dimension: {latent.shape[1]}")


# -----------------------------
# Load cell types
# -----------------------------
with open(cell_type_path, "r") as f:
    cell_types = np.array([line.strip() for line in f])

assert len(cell_types) == latent.shape[0], (
    f"Number of labels ({len(cell_types)}) does not match "
    f"number of latent embeddings ({latent.shape[0]})"
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

colors = np.array([label_to_color[label] for label in cell_types])


# -----------------------------
# Helper
# -----------------------------
def remove_axis_numbers(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])


# -----------------------------
# UMAP plot
# -----------------------------
def draw_umap(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42,
    )

    embedding = reducer.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(9, 7))

    for label in unique_cell_types:
        mask = cell_types == label
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

    legend_handles = [
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

    ax.legend(
        handles=legend_handles,
        title="Cell Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    title = f"PeakVI UMAP, n_neighbors={n_neighbors}, min_dist={min_dist}"
    ax.set_title(title)

    plt.tight_layout()

    filename = f"{name}_umap_nn{n_neighbors}_mindist{min_dist}.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.show()


# -----------------------------
# Generate plots
# -----------------------------
for n_neighbors in [5, 15, 50]:
    for min_dist in [0.1, 0.5, 0.9]:
        draw_umap(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )