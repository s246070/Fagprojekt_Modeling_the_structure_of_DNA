import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

data = torch.load("models/peakvi_latent_2d.pth")

neigh = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
neigh.fit(data, cell_types)
pred = neigh.predict(data)

print(f"Accuracy: {accuracy_score(cell_types, pred)}, NMI: {normalized_mutual_info_score(cell_types, pred)}, ARI: {adjusted_rand_score(cell_types, pred)}")