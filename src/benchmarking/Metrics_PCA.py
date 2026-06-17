import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

embed_data = np.load("models/pca_full_embeddings.npy")

for k in [2, 3, 8, 16, 32]:
    data = embed_data[:, :k]
    for i in [15, 30, 50]:
        nbrs = NearestNeighbors(n_neighbors=i + 1, metric='euclidean').fit(data)
        distances, indices = nbrs.kneighbors(data)

        # skip first neighbor (itself)
        indices = indices[:, 1:]

        preds = []
        for _, neighbors in enumerate(indices):
            neighbor_labels = [cell_types[j] for j in neighbors]
            pred = max(set(neighbor_labels), key=neighbor_labels.count)
            preds.append(pred)

        print(f"LOO-style accuracy (dimensionality={k}, neighbors={i}): {accuracy_score(cell_types, preds)}, NMI: {normalized_mutual_info_score(cell_types, preds)}, ARI: {adjusted_rand_score(cell_types, preds)}")