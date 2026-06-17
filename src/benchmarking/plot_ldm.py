import torch
import matplotlib.pyplot as plt
import numpy as np

name = "ldm_ls2_epoch1000_blocks100_index1"

model = torch.load(f"models/{name}.pth")
data = model['embed_cells'].cpu().detach().numpy()
data_2 = model['embed_features'].cpu().detach().numpy()

data = np.concatenate((data_2, data), axis=0)

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]

labels = np.array(cell_types)
unique_labels = np.unique(labels)

unique_labels = np.append(unique_labels, "Features") 
labels = np.append(["Features"] * data_2.shape[0], labels)

color_map = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))
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
plt.savefig(f"plots/full_{name}.png", dpi=300)
plt.show()