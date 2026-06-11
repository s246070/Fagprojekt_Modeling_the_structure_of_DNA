import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

name = "ldm_ls2_weighting_False_run4000"

model = torch.load(f"models/{name}.pth")
data = model['embed_cells'].cpu().detach().numpy()

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

labels = np.array(cell_types)
unique_labels = np.unique(labels)

color_map = plt.cm.get_cmap("tab10", len(unique_labels))
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

colors = [label_to_color[label] for label in labels]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_color[label], label=label)
    for label in unique_labels
]
plt.legend(handles=handles, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f"plots/{name}.png", dpi=300)
plt.show()