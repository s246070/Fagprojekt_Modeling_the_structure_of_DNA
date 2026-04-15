print("starting main.py")
from data import Data
print("imported data")
from model import LDM
print("imported model")
from train import TrainModel
print("imported train")
from evaluate import make_test_set, evaluate
print("imported evaluate")
import torch
print("imported torch")
import matplotlib.pyplot as plt
#import os
import os
# get latent space dimension from environment variable in main.lsf, default to 8 if not set
ls_dim = int(os.getenv("LS_DIM", 2))

print("everything is imported!")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True)

print("data added")
# Convert AnnData.X to tensor
Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

Aij, targets = make_test_set(Aij, percentage=0.1)

# Initialize model
model = LDM(
    data=Aij,
    ls_dim=ls_dim,
    device=device,
    epochs=6_001,
    lr=1e-3,
    seed=42
)

print("beginning training")
# Train
losses = TrainModel(model, device=device, threads=6)
print("training complete")

auc, auc_data = evaluate(model, Aij, targets)
print(f"AUROC: {auc:.4f}")

# print("plotting")
# # cell_embeddings = model.embed_cells.detach().cpu().numpy()

# plt.figure(figsize=(7, 6))
# plt.scatter(cell_embeddings[:, 0], cell_embeddings[:, 1], s=5, alpha=0.7)
# plt.xlabel("Latent dim 1")
# plt.ylabel("Latent dim 2")
# plt.title("Cell latent space")
# plt.savefig("latent_space_10000.png", dpi=300)
