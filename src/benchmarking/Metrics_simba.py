import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]

adata = sc.read_h5ad("models/adata_cells_ldm32_simba.h5ad")

data = adata.X

for i in [15, 30, 50]:
    neigh = KNeighborsClassifier(n_neighbors=i + 1, metric="euclidean")
    neigh.fit(data, cell_types)
    pred = neigh.predict(data)

    print(
        f"Accuracy: {accuracy_score(cell_types, pred)}, "
        f"NMI: {normalized_mutual_info_score(cell_types, pred)}, "
        f"ARI: {adjusted_rand_score(cell_types, pred)}"
    )