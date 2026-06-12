import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

model = torch.load("models/ldm_ls2_weighting_False_run100.pth")

data = model['embed_cells'].cpu().detach().numpy()

neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
neigh.fit(data, cell_types)
pred = neigh.predict(data)

print(accuracy_score(cell_types, pred))