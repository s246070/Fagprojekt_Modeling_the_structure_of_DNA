print("starting PeakVI main_peakvi.py", flush=True)

import os
import random
import sys
from pathlib import Path
import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

print("imports done", flush=True)

from src.dna_modelling.data import Data
import scvi
from scvi.model import PEAKVI
from src.PeakVI.evaluate_peakvi import evaluate_peakvi
from src.dna_modelling.evaluate import make_test_set

print("modules loaded", flush=True)

# config from env
max_epochs = int(os.getenv("EPOCHS", "200"))
index = int(os.getenv("INDEX", "1"))
seed = int(os.getenv("SEED", str(index)))
model_dir = os.getenv("MODEL_DIR", f"models/peakvi_run{index}")

# reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"seed: {seed}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True)  # Load in backed mode to save memory
print("adata loaded", flush=True)
print(type(adata.X), flush=True)

# Create test set once before training
print("Creating test set...", flush=True)
from src.PeakVI.evaluate_peakvi import _to_dense_numpy
import scipy.sparse as sp
A = _to_dense_numpy(adata.X)
device_tensor = torch.device("cpu")
A_tensor = torch.tensor(A, dtype=torch.int8, device=device_tensor)
A_mod, test_targets, test_target_zeros = make_test_set(A_tensor, percentage=0.1)

# Switch to an in-memory AnnData before mutating X.
if getattr(adata, "isbacked", False):
	adata = adata.to_memory()
	print("Converted backed adata to in-memory copy", flush=True)

# Remove test targets from training data
X = adata.X
if sp.issparse(X):
	X = X.tolil()
	for row, col in test_targets:
		X[row, col] = 0
	adata.X = X.tocsr()
	print("Test targets removed from sparse adata.X", flush=True)
else:
	for row, col in test_targets:
		X[row, col] = 0
	adata.X = X
	print("Test targets removed from dense adata.X", flush=True)

# Setup PeakVI with modified adata
scvi.model.PEAKVI.setup_anndata(adata)
model = scvi.model.PEAKVI(adata)

# Training loop in intervals so we can evaluate and save metrics periodically
interval = int(os.getenv("INTERVAL", "50"))
os.makedirs("results", exist_ok=True)
results_path = f"results/peakvi_run{index}.csv"
with open(results_path, "w") as rf:
	rf.write("Epoch,AUROC,AUPR,F1max\n")

current_epoch = 0

while current_epoch < max_epochs:
	to_train = min(interval, max_epochs - current_epoch)
	print(f"Training epochs {current_epoch+1}..{current_epoch+to_train}", flush=True)
	model.train(max_epochs=to_train)
	current_epoch += to_train

	print("evaluating model...", flush=True)
	metrics = evaluate_peakvi(model, adata, targets=test_targets, target_zeros=test_target_zeros)

	# append metrics to results CSV
	with open(results_path, "a") as rf:
		rf.write(f"{current_epoch},{metrics['auroc']:.6f},{metrics['aupr']:.6f},{metrics['f1_max']:.6f}\n")

	print(f"Epoch {current_epoch}: AUROC={metrics['auroc']:.4f}, AUPR={metrics['aupr']:.4f}, F1max={metrics['f1_max']:.4f}")

# Save final model
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
model.save(model_dir, overwrite=True)
print(f"model saved to {model_dir}", flush=True)

sys.exit()
