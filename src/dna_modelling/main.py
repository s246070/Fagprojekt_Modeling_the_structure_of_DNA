print("starting main.py", flush=True)

import torch
import matplotlib.pyplot as plt
import os
import sys

print("imported all libraries", flush=True)

from data import Data
from model import LDM
from train import TrainModel
from evaluate import *

print("imported all files", flush=True)

ls_dim = int(os.getenv("LS_DIM", 2))

print("everything is imported!", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device, flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True)

print("data added", flush=True)

# Convert AnnData.X to tensor
Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

Aij, targets, target_zeros = make_test_set(Aij, percentage=0.1)

# Initialize model
model = LDM(
    data=Aij,
    ls_dim=ls_dim,
    device=device,
    epochs=6_001,
    lr=1e-3,
    seed=42
)

print("beginning training", flush=True)

# Train
losses = TrainModel(model, device=device, plots=False, targets=targets, target_zeros=target_zeros)

print("training complete", flush=True)

auc, auc_data, f1_score = validate(model, Aij, targets, target_zeros)

#save model in models folder
LDM.save_model(model, f"models/ldm_ls{ls_dim}.pth")

print(f"AUROC: {auc:.4f}", flush=True)
print(f"F1 Score: {f1_score:.4f}", flush=True)

#stop job when done
sys.exit(0)