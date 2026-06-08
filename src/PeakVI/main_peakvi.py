print("starting PeakVI main_peakvi.py", flush=True)

import os
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

print("imports done", flush=True)

from src.dna_modelling.data import Data
import scvi
from scvi.model import PEAKVI
from src.PeakVI.evaluate_peakvi import evaluate_peakvi

print("modules loaded", flush=True)

# config from env
max_epochs = int(os.getenv("EPOCHS", "200"))
index = int(os.getenv("INDEX", "1"))
model_dir = os.getenv("MODEL_DIR", f"models/peakvi_run{index}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True)
print("adata loaded", flush=True)

# Setup PeakVI
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
	metrics = evaluate_peakvi(model, adata, test_fraction=0.1)

	# append metrics to results CSV
	with open(results_path, "a") as rf:
		rf.write(f"{current_epoch},{metrics['auroc']:.6f},{metrics['aupr']:.6f},{metrics['f1_max']:.6f}\n")

	print(f"Epoch {current_epoch}: AUROC={metrics['auroc']:.4f}, AUPR={metrics['aupr']:.4f}, F1max={metrics['f1_max']:.4f}")

# Save final model
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
model.save(model_dir, overwrite=True)
print(f"model saved to {model_dir}", flush=True)

sys.exit()
