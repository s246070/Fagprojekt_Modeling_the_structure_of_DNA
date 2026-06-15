import torch
from dna_modelling.utils import Data
from dna_modelling.evaluate import make_test_set
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import os

device = "cpu"

print(f"everything is imported! {datetime.now()}, device: {device}", flush=True)

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=True)

# Important when copying backed AnnData
adata = adata.to_memory()

for i in range(1, 6):
    rng = np.random.default_rng(seed=i)

    idx = rng.choice(adata.n_obs, size=10_000, replace=False)
    adata_10k = adata[idx].copy()

    target_n_peaks = 200_000

    if adata_10k.n_vars > target_n_peaks:
        n_cells_per_peak = np.asarray((adata_10k.X > 0).sum(axis=0)).ravel()
        top_idx = np.argsort(n_cells_per_peak)[-target_n_peaks:]
        adata_10k = adata_10k[:, top_idx].copy()
    
    adata_10k.write_h5ad(f"data/train_sets/adata_subset_10k_{i}.h5ad")