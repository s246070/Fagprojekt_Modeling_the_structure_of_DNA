from scvi_colab import install
import os
import tempfile
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import scvi
import seaborn as sns
import torch

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

# load data from root folder
adata_path = "/Users/andreasstampedalgaard/Fagprojekt_Modeling_the_structure_of_DNA/data/lung_atlas_preprocessed.h5ad"
adata = sc.read(
    adata_path,
    backup_url="https://figshare.com/ndownloader/files/52859288",
)

query_mask = np.array([s in ["AT1", "AT2"] for s in adata.obs["leiden_cluster"]])

adata_ref = adata[~query_mask].copy()
adata_query = adata[query_mask].copy()

sc.pp.highly_variable_genes(adata_ref, n_top_genes=2000, batch_key="leiden_cluster", subset=True)
adata_query = adata_query[:, adata_ref.var_names].copy()

scvi.data.setup_anndata(adata_ref, batch_key="leiden_cluster", layer="X")

scvi_ref = scvi.model.SCVI(
    adata_ref,
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
)

scvi_ref.train()