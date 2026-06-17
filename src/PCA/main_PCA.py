from pathlib import Path
import pickle
import sys

import numpy as np
import scanpy as sc

parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from dna_modelling.utils import Data

sc.settings.verbosity = 2

adata = Data().load_data(backed=False, full=True)
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

adata_pca = adata.copy()
sc.pp.normalize_total(adata_pca, target_sum=1e4)
sc.pp.log1p(adata_pca)
sc.pp.scale(adata_pca, max_value=10)
sc.tl.pca(adata_pca, n_comps=50, svd_solver="arpack")

pca_embeddings = adata_pca.obsm["X_pca"]
np.save(output_dir / "pca_full_embeddings.npy", pca_embeddings)
np.save(output_dir / "pca_full_components.npy", adata_pca.varm["PCs"])
np.save(output_dir / "pca_full_variance_ratio.npy", adata_pca.uns["pca"]["variance_ratio"])
with open(output_dir / "pca_full_model.pkl", "wb") as handle:
	pickle.dump(adata_pca.uns["pca"], handle)
adata_pca.write_h5ad(output_dir / "pca_full_dataset.h5ad")

print(f"Saved PCA embeddings and components to {output_dir.resolve()}")
