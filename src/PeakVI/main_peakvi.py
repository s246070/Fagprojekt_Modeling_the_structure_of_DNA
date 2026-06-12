import scanpy as sc
import scvi
import torch
import sys
from pathlib import Path
import scipy.sparse as sp

# Add parent directory: project/
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from dna_modelling.data import Data

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# -----------------------------
# 1. Load AnnData
# -----------------------------
# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=False, full=True)

if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
    adata.X = adata.X.tocsr()


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
    n_latent=16
)

model.train(
    max_epochs=200,
    accelerator="cpu",
    devices=1,
    batch_size=8192,
    early_stopping=True,
    early_stopping_patience=20,
    datasplitter_kwargs={"num_workers": 0},
)


# -----------------------------
# 4. Extract latent embeddings
# -----------------------------
latent_16d = model.get_latent_representation()

print(latent_16d.shape)
# (n_cells, 16)


# -----------------------------
# 5. Convert to PyTorch tensor
# -----------------------------
latent_tensor = torch.tensor(
    latent_16d,
    dtype=torch.float32
)

print(latent_tensor.shape)
# torch.Size([n_cells, 16])


# -----------------------------
# 6. Save as .pth file
# -----------------------------
torch.save(latent_tensor, "models/peakvi_latent_16d.pth")
torch.save(
    {
        "latent": latent_tensor,
        "cell_ids": list(adata.obs_names),
        "n_latent": 16,
    },
    "models/peakvi_latent_16d_extra.pth"
)