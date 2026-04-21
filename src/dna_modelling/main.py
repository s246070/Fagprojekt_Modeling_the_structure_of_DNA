from data import Data
from model import LDM
from train import TrainModel
import torch
import matplotlib.pyplot as plt

print("everything is imported!")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True)

print("data added")
# Convert AnnData.X to tensor
Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

# Initialize model
model = LDM(
    data=Aij,
    ls_dim=2,
    device=device,
    epochs=10_001,
    lr=1e-3,
    seed=42
)

print("beginning training")
# Train
losses = TrainModel(model, device=device, threads=6)

print("plotting")
cell_embeddings = model.embed_cells.detach().cpu().numpy()

plt.figure(figsize=(7, 6))
plt.scatter(cell_embeddings[:, 0], cell_embeddings[:, 1], s=5, alpha=0.7)
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.title("Cell latent space")
plt.savefig("latent_space_10000.png", dpi=300)

model.save_model(f"ldm_model_{ls_dim}.pth")