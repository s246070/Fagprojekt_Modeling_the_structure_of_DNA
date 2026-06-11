import torch
import matplotlib.pyplot as plt
import os
import sys

from data import Data
from model import LDM
from train import TrainModel
from evaluate import *
from datetime import datetime

seed = int(os.getenv("SEED", 42))

ls_dim = int(os.getenv("LS_DIM", 2))

index = int(os.getenv("INDEX", 1))

weighting = os.getenv("WEIGHTING", "false").lower() == "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"everything is imported!{datetime.now()}, device: {device}", flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=True)

print(f"data added{datetime.now()}", flush=True)

# Convert AnnData.X to tensor
Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

print(f"converted data to tensor{datetime.now()}", flush=True)

Aij, targets, target_zeros = make_test_set(Aij, percentage=0.1)

print(f"created test set{datetime.now()}", flush=True)

# Initialize model
model = LDM(
    data=Aij,
    ls_dim=ls_dim,
    device=device,
    epochs=1001,
    lr=0.03,
    weighting=weighting,
    index=index,
    seed=seed
)

print(f"beginning training{datetime.now()}", flush=True)

# Train
losses = TrainModel(model, device=device, plots=False, targets=targets, target_zeros=target_zeros)

print(f"training complete{datetime.now()}", flush=True)

auc, auc_data, f1_score, pr_auc, pr_curve_data = validate(model, Aij, targets, target_zeros)

print(f"AUROC: {auc:.4f}", flush=True)
print(f"F1 Score: {f1_score:.4f}", flush=True)
print(f"PR AUC: {pr_auc:.4f}", flush=True)

#stop job when done
sys.exit()