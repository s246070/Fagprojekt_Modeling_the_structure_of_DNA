import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

adata = sc.read_h5ad("results/simba/simba_subset_dim{2}/adata_cells_ldm2_simba.h5ad")

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]


labels = np.array(cell_types)
unique_labels = np.unique(labels)

color_map = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

colors = [label_to_color[label] for label in labels]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(adata.X[:, 0], adata.X[:, 1], c=colors, alpha=0.7)
handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_color[label], label=label)
    for label in unique_labels
]
plt.legend(handles=handles, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f"plots/simba_latentspace.png", dpi=300)
plt.show()