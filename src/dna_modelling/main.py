print("starting main.py")

import torch
import matplotlib.pyplot as plt
import os

print("imported all libraries")

from data import Data
from model import LDM
from train import TrainModel
from evaluate import make_test_set, evaluate

print("imported all files")

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
losses = TrainModel(model, device=device, plots=False)

print("training complete")

auc, auc_data = evaluate(model, Aij, targets)

print(f"AUROC: {auc:.4f}")
