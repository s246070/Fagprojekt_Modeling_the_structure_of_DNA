import torch
from dna_modelling.utils import Data
from dna_modelling.evaluate import make_test_set
from datetime import datetime
import numpy as np
import scipy.sparse as sp


def save_tensor_as_anndata(Aij_temp, original_adata, targets, target_zeros, save_path):
    # Build a sparse matrix directly from the nonzero coordinates.
    # This avoids densifying the tensor into an intermediate NumPy array.
    Aij_cpu = Aij_temp.detach().cpu()
    row_idx, col_idx = Aij_cpu.nonzero(as_tuple=True)
    values = np.ones(row_idx.numel(), dtype=np.float32)
    X = sp.csr_matrix((values, (row_idx.numpy(), col_idx.numpy())), shape=Aij_cpu.shape)

    # Copy original AnnData metadata
    adata_out = original_adata.copy()

    # Replace matrix with modified training matrix
    adata_out.X = X

    # Store test-set information
    adata_out.uns["targets"] = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else targets
    )

    adata_out.uns["target_zeros"] = (
        target_zeros.detach().cpu().numpy()
        if isinstance(target_zeros, torch.Tensor)
        else target_zeros
    )

    # Save as h5ad
    adata_out.write_h5ad(save_path)

    return adata_out


# This script only prepares train sets, so keep the matrix on CPU to avoid
# loading the full tensor into GPU memory.
device = "cpu"

print(f"everything is imported! {datetime.now()}, device: {device}", flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=True)

# Important when copying backed AnnData
adata = adata.to_memory()

print(f"data added {datetime.now()}", flush=True)

# Convert AnnData.X to tensor
Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

print(f"converted data to tensor {datetime.now()}", flush=True)

for seed in range(2, 6):
    print(f"beginning seed {seed} {datetime.now()}", flush=True)

    torch.manual_seed(seed)

    Aij_temp, targets, target_zeros = make_test_set(Aij, percentage=0.1)

    save_tensor_as_anndata(
        Aij_temp=Aij_temp,
        original_adata=adata,
        targets=targets,
        target_zeros=target_zeros,
        save_path=f"data/train_sets/Aij_train_{seed}.h5ad"
    )

    print(f"finished seed {seed} {datetime.now()}", flush=True)