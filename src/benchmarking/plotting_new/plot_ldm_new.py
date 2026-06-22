import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib.lines import Line2D

# -----------------------------
# Settings
# -----------------------------
name = "ldm_ls2_weighting_False_run100"
model_path = f"models/plotting/{name}.pth"
cell_type_path = "src/benchmarking/cell_types.txt"
save_dir = Path("plots")
save_dir.mkdir(exist_ok=True)


# -----------------------------
# Load model embeddings
# -----------------------------
model = torch.load(model_path, map_location="cpu")

cell_embeddings = model["embed_cells"].detach().cpu().numpy()
feature_embeddings = model["embed_features"].detach().cpu().numpy()

with open(cell_type_path, "r") as f:
    cell_types = np.array([line.strip() for line in f])

assert len(cell_types) == cell_embeddings.shape[0], (
    f"Number of labels ({len(cell_types)}) does not match "
    f"number of cell embeddings ({cell_embeddings.shape[0]})"
)

latent_dim = cell_embeddings.shape[1]
print(f"Loaded {cell_embeddings.shape[0]} cells")
print(f"Loaded {feature_embeddings.shape[0]} features")
print(f"Latent dimension: {latent_dim}")


# -----------------------------
# Colors
# -----------------------------
unique_cell_types = np.unique(cell_types)

cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cell_types))
label_to_color = {
    label: cmap(i)
    for i, label in enumerate(unique_cell_types)
}

feature_color = "black"


# -----------------------------
# Helper: remove axis numbers
# -----------------------------
def remove_axis_numbers(ax, is_3d=False):
    # Keep tick positions so the grid can be drawn,
    # but hide the actual numbers.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if is_3d:
        ax.set_zticklabels([])


# -----------------------------
# 2D plot
# -----------------------------
def plot_ldm_2d():
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot all features first, behind cells
    ax.scatter(
        feature_embeddings[:, 0],
        feature_embeddings[:, 1],
        s=0.5,
        c=feature_color,
        alpha=0.05,
        rasterized=True
    )

    # Plot all cells by cell type
    for label in unique_cell_types:
        mask = cell_types == label
        ax.scatter(
            cell_embeddings[mask, 0],
            cell_embeddings[mask, 1],
            s=2,
            color=label_to_color[label],
            alpha=0.75,
            rasterized=True
        )

    ax.set_xlim(-3.75, 3.75)
    ax.set_ylim(-3.75, 3.75)

    ax.set_xlabel("LDM1")
    ax.set_ylabel("LDM2")

    remove_axis_numbers(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend handles
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5,
            label="Features"
        )
    ]

    for label in unique_cell_types:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=label_to_color[label],
                markeredgecolor=label_to_color[label],
                markersize=6,
                label=label
            )
        )

    # ax.legend(
    #     handles=legend_handles,
    #     title="Cell Types",
    #     bbox_to_anchor=(1.02, 1),
    #     loc="upper left",
    #     frameon=False
    # )

    plt.tight_layout()
    plt.savefig(save_dir / f"{name}_2d.png", dpi=600, bbox_inches="tight")
    plt.show()


# -----------------------------
# 3D plot
# -----------------------------
def plot_ldm_3d():
    if latent_dim < 3:
        print(
            "Skipping 3D plot because the model only has "
            f"{latent_dim} latent dimensions. Train/load an LDM with ls_dim >= 3."
        )
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all features first, behind cells
    ax.scatter(
        feature_embeddings[:, 0],
        feature_embeddings[:, 1],
        feature_embeddings[:, 2],
        s=0.5,
        c=feature_color,
        alpha=0.2,
        label="Features",
        rasterized=True
    )

    # Custom legend handles
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5,
            label="Features"
        )
    ]

    for label in unique_cell_types:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=label_to_color[label],
                markeredgecolor=label_to_color[label],
                markersize=6,
                label=label
            )
        )

    # Plot all cells by cell type
    for label in unique_cell_types:
        mask = cell_types == label
        ax.scatter(
            cell_embeddings[mask, 0],
            cell_embeddings[mask, 1],
            cell_embeddings[mask, 2],
            s=2,
            color=label_to_color[label],
            alpha=0.85,
            rasterized=True
        )

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2)
    ax.set_zlim(-3.5, 3.5)

    ax.set_xlabel("LDM1", labelpad = 0)
    ax.set_ylabel("LDM2", labelpad= 0)
    ax.set_zlabel("LDM3", labelpad= 0)

    ax.grid(True)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    
    remove_axis_numbers(ax, is_3d=True)

    ax.view_init(elev=20, azim=130)

    # ax.legend(
    #     handles=legend_handles,
    #     title="Cell Types",
    #     bbox_to_anchor=(1.1, 1),
    #     loc="upper left",
    #     frameon=False
    # )

    #configure layout
    plt.subplots_adjust(left=-0.1, right=1.05, bottom=-0.1, top=1.1)
    # plt.tight_layout()

    plt.savefig(save_dir / f"{name}_3d.png", dpi=600)
    plt.show()


plot_ldm_2d()