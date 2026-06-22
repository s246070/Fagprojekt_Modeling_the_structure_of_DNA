import scvi
import torch
import sys
from pathlib import Path
import scipy.sparse as sp

# Add parent directory: project/
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from dna_modelling.utils import Data  # noqa: E402

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# -----------------------------
# 1. Load AnnData
# -----------------------------
# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=False, specify_path="train_sets/adata_subset_10k_1.h5ad")

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
    accelerator="gpu",
    devices=1,
    early_stopping=True,
    batch_size=128,
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
torch.save(latent_tensor, f"models/peakvi_latent_{latent_tensor.shape[1]}d_subset_10k.pth")