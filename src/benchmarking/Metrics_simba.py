import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

embedding_path = "results/simba/simba_subset_dim2/adata_cells_simba.h5ad"

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

adata = sc.read_h5ad(embedding_path)
data = adata.X

nbrs = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(data)
distances, indices = nbrs.kneighbors(data)

# Skip first neighbor because it is the cell itself
indices = indices[:, 1:]

preds = []
for neighbors in indices:
    neighbor_labels = [cell_types[j] for j in neighbors]
    pred = max(set(neighbor_labels), key=neighbor_labels.count)
    preds.append(pred)

print(
    f"LOO-style accuracy: {accuracy_score(cell_types, preds)}, "
    f"NMI: {normalized_mutual_info_score(cell_types, preds)}, "
    f"ARI: {adjusted_rand_score(cell_types, preds)}"
)