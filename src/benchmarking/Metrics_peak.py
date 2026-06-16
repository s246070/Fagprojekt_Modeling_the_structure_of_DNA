import torch
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

data = torch.load("models/peakvi_latent_2d_subset_10k.pth")

for i in [15, 30, 50]:
    nbrs = NearestNeighbors(n_neighbors=i + 1, metric='euclidean').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # skip first neighbor (itself)
    indices = indices[:, 1:]

    preds = []
    for i, neighbors in enumerate(indices):
        neighbor_labels = [cell_types[j] for j in neighbors]
        pred = max(set(neighbor_labels), key=neighbor_labels.count)
        preds.append(pred)

    print(f"LOO-style accuracy: {accuracy_score(cell_types, preds)}, NMI: {normalized_mutual_info_score(cell_types, preds)}, ARI: {adjusted_rand_score(cell_types, preds)}")