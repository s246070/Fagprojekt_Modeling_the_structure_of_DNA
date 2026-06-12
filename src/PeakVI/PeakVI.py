import os
import scanpy as sc
import scvi

adata_path = os.path.join("data", "lung_atlas_preprocessed.h5ad")
adata_path
print(os.getcwd())

# load data from root folder
adata_path = "/Users/andreasstampedalgaard/Fagprojekt_Modeling_the_structure_of_DNA/data/lung_atlas_preprocessed.h5ad"
adata = sc.read(
    adata_path,
    backup_url="https://figshare.com/ndownloader/files/52859288",
)

scvi.model.PEAKVI.setup_anndata(adata)
model = scvi.model.PEAKVI(adata)
model.train(accelerator='cpu') # use accelerator='cuda' if you have a GPU available, 'mps' on macOS or 'cpu' otherwise

# Retrieve the latent representation and store it in adata.obsm

PEAKVI_LATENT_KEY = "X_peakvi"

latent = model.get_latent_representation()
adata.obsm[PEAKVI_LATENT_KEY] = latent
latent.shape

PEAKVI_CLUSTERS_KEY = "clusters_peakvi"

# compute the k-nearest-neighbor graph that is used in both clustering and umap algorithms
sc.pp.neighbors(adata, use_rep=PEAKVI_LATENT_KEY)
# compute the umap
sc.tl.umap(adata, min_dist=0.2)
# cluster the space (we use a lower resolution to get fewer clusters than the default)
sc.tl.leiden(adata, key_added=PEAKVI_CLUSTERS_KEY, resolution=0.2)

model_dir = os.path.join("models", "peakvi_pbmc")
model.save(model_dir, overwrite=True)