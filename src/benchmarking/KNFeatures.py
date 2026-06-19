import torch
from sklearn.cluster import KMeans
import numpy as np
import math
import scanpy as sc

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]

model = torch.load("models/ldm_ls2_epoch1000_index1.pth")

adata = sc.read_h5ad("data/train_sets/adata_subset_10k_1.h5ad", backed="r")

data = model['embed_cells'].cpu().detach().numpy()
features = model['embed_features'].cpu().detach().numpy()

types_pos = {i: [] for i in set(cell_types)}

for i in range(len(data)):
    types_pos[cell_types[i]].append(i)

centroids = {}

for cell_type, positions in types_pos.items():
    centroids[cell_type] = np.mean(data[positions], axis=0)

closest_features = {i: [] for i in set(cell_types)}

for cell_type, centroid in centroids.items():
    distances = np.linalg.norm(features - centroid, axis=1)
    closest_indices = np.argsort(distances)[:50]
    closest_features[cell_type] = adata.var.Feature[closest_indices].tolist()

with open(f"results/KNFeatures_no_batching.csv", "w") as f:
    for i in closest_features.keys():
        f.write(f"{i},")
    f.write("\n")
    for i in range(50):
        for cell_type in closest_features.keys():
            f.write(f"{closest_features[cell_type][i]}," if i < len(closest_features[cell_type]) else ",")
        f.write("\n")

with open(f"results/KNFeatures_no_batching.txt", "w") as f:
    f.write(closest_features.__str__())