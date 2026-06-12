from pathlib import Path
from urllib.request import urlretrieve
import scanpy as sc
import scipy.sparse as sp
import torch

"""
This class is responsible for downloading and loading the data. 
It ensures that the data is downloaded to a specific directory 
and provides a method to load the data into either memory or hard drive.
"""
class Data():
    """ 
    The constructor initializes the data directory, the path to the data file, 
    and the backup URL for downloading the data if it is not already present. 
    """
    def __init__(self):
        self.data_directory = Path("data")
        self.data_path = self.data_directory / "lung_atlas_preprocessed.h5ad"
        self.data_path_full = self.data_directory / "hematopoiesis_GSE129785_FACS_sorted.h5ad"
        self.backup_url = "https://figshare.com/ndownloader/files/52859288"

    """
    This method checks if the data file exists at the specified path. 
    If it does not exist, it creates the necessary directories and downloads the data from the backup URL.
    """
    def ensure_data_downloaded(self, path=None):
        if path is None:
            path = self.data_directory

        if not path.exists():
            print(f"Downloading datset to {self.data_path}...")
            urlretrieve(self.backup_url, self.data_path)

    """
    This method loads the data from the specified path. 
    If the 'backed' parameter is set to True, it loads the data into the hard drive.
    If 'backed' is False, it loads the entire dataset into memory.
    """
    def load_data(self, backed=True, full=False):
        self.ensure_data_downloaded(path=self.data_path_full if full else self.data_path)

        if backed:
            return sc.read_h5ad(self.data_path, backed="r") if not full else sc.read_h5ad(self.data_path_full, backed="r")
        return sc.read_h5ad(self.data_path) if not full else sc.read_h5ad(self.data_path_full)
    
    def load_test_set(self, path, device):
        adata = sc.read_h5ad(path)
        Aij = self.anndata_to_tensor(adata, device=device, make_binary=True)
        targets = adata.uns["targets"]
        target_zeros = adata.uns["target_zeros"]
        return Aij, targets, target_zeros



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



if __name__ == "__main__":
    data = Data()
    adata = data.load_data(backed=True)
    print(adata)