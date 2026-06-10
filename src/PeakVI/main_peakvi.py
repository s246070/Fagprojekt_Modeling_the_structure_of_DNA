import scanpy as sc
import scvi
import torch
import sys
from pathlib import Path

# Add parent directory: project/
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from dna_modelling.data import Data

# -----------------------------
# 1. Load AnnData
# -----------------------------
# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=True)  # Load in backed mode to save memory


# -----------------------------
# 2. Register AnnData for PeakVI
# -----------------------------
scvi.model.PEAKVI.setup_anndata(
    adata
)


# -----------------------------
# 3. Train PeakVI with 2 latent dimensions
# -----------------------------
model = scvi.model.PEAKVI(
    adata,
    n_latent=2
)

model.train(
    max_epochs=1000,
    accelerator="auto"
)


# -----------------------------
# 4. Extract latent embeddings
# -----------------------------
latent_2d = model.get_latent_representation()

print(latent_2d.shape)
# (n_cells, 2)


# -----------------------------
# 5. Convert to PyTorch tensor
# -----------------------------
latent_tensor = torch.tensor(
    latent_2d,
    dtype=torch.float32
)

print(latent_tensor.shape)
# torch.Size([n_cells, 2])


# -----------------------------
# 6. Save as .pth file
# -----------------------------
torch.save(latent_tensor, "models/peakvi_latent_2d.pth")
torch.save(
    {
        "latent": latent_tensor,
        "cell_ids": list(adata.obs_names),
        "n_latent": 2,
    },
    "models/peakvi_latent_2d_extra.pth"
)