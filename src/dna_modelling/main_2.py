import torch
import os
import sys

from data import Data
from model import LDM
from train import TrainModel
from evaluate import *
from datetime import datetime

seed = int(os.getenv("SEED", 1))

ls_dim = int(os.getenv("LS_DIM", 2))

index = int(os.getenv("INDEX", 1))

weighting = os.getenv("WEIGHTING", "false").lower() == "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"everything is imported!{datetime.now()}, device: {device}", flush=True)

# Load Aij, targets, and target_zeros from data/train_sets
Aij, targets, target_zeros = Data().load_test_set(f"data/train_sets/Aij_train_{seed}.h5ad", device=device)

print(f"converted data to tensor{datetime.now()}", flush=True)


# Initialize model
model = LDM(
    data=Aij,
    ls_dim=ls_dim,
    device=device,
    epochs=1001,
    lr=0.05,
    weighting=weighting,
    index=index,
    seed=seed
)

print(f"beginning training{datetime.now()}", flush=True)

# Train
losses = TrainModel(model, device=device, plots=False, targets=targets, target_zeros=target_zeros, batching=True, num_blocks=100)

print(f"training complete{datetime.now()}", flush=True)

auc, auc_data, f1_score, pr_auc, pr_curve_data = validate(model, Aij, targets, target_zeros)

print(f"AUROC: {auc:.4f}", flush=True)
print(f"F1 Score: {f1_score:.4f}", flush=True)
print(f"PR AUC: {pr_auc:.4f}", flush=True)

#stop job when done
sys.exit()