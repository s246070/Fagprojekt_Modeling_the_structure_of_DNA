from pathlib import Path
from urllib.request import urlretrieve
import scanpy as sc
import scipy.sparse as sp
import torch

class Data():

    def __init__(self):
        self.data_directory = Path("data")
        self.data_path = self.data_directory / "lung_atlas_preprocessed.h5ad"
        self.backup_url = "https://figshare.com/ndownloader/files/52859288"

    def ensure_data_downloaded(self):
        self.data_directory.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            print(f"Downloading datset to {self.data_path}...")
            urlretrieve(self.backup_url, self.data_path)

    def load_data(self, backed=False):
        self.ensure_data_downloaded()

        if backed:
            return sc.read_h5ad(self.data_path, backed="r")
        return sc.read_h5ad(self.data_path)

    def anndata_to_tensor(self, adata, device, make_binary=True):
        X = adata.X

        # Load into memory if backed/lazy
        if hasattr(X, "__getitem__") and not isinstance(X, (list, tuple)):
            try:
                X = X[:]
            except Exception:
                pass

        # Convert sparse to dense
        if sp.issparse(X):
            X = X.toarray()

        # Convert to tensor
        X = torch.as_tensor(X, dtype=torch.float32, device=device)

        if make_binary:
            X = (X > 0).float()

        return X