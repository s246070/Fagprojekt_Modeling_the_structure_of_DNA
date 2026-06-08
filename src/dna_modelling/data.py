from pathlib import Path
from urllib.request import urlretrieve
import scanpy as sc
import scipy.sparse as sp
import numpy as np
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
    
    def remove_zero_rows_and_columns(self, X):
        """Remove rows and columns that are all zeros from X.

        Supports scipy sparse matrices, numpy arrays, and torch tensors.
        Returns the filtered array/matrix of the same type (for sparse and numpy)
        or a torch tensor for tensor inputs.
        """
        # scipy sparse
        if sp.issparse(X):
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            col_sums = np.asarray(X.sum(axis=0)).ravel()
            non_zero_row_mask = row_sums > 0
            non_zero_col_mask = col_sums > 0
            return X[non_zero_row_mask][:, non_zero_col_mask]

        # numpy array
        if isinstance(X, np.ndarray):
            non_zero_row_mask = X.sum(axis=1) > 0
            non_zero_col_mask = X.sum(axis=0) > 0
            return X[non_zero_row_mask][:, non_zero_col_mask]

        # torch tensor
        if isinstance(X, torch.Tensor):
            row_sums = X.sum(dim=1)
            col_sums = X.sum(dim=0)
            non_zero_row_mask = (row_sums > 0).to(torch.bool)
            non_zero_col_mask = (col_sums > 0).to(torch.bool)
            X = X[non_zero_row_mask, :]
            X = X[:, non_zero_col_mask]
            return X

        # Fallback: try array-like interface
        try:
            non_zero_row_mask = X.sum(axis=1) > 0
            non_zero_col_mask = X.sum(axis=0) > 0
            return X[non_zero_row_mask][:, non_zero_col_mask]
        except Exception:
            raise TypeError("Unsupported type for remove_zero_rows_and_columns")


if __name__ == "__main__":
    data = Data()
    adata = data.load_data(backed=True, full=True)

    # Compute non-zero row/col masks without converting the whole matrix to a dense tensor
    X = adata.X
    if sp.issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        col_sums = np.asarray(X.sum(axis=0)).ravel()
        non_zero_row_mask = row_sums > 0
        non_zero_col_mask = col_sums > 0
    else:
        try:
            Xarr = X[:] if hasattr(X, "__getitem__") else X
        except Exception:
            Xarr = X
        non_zero_row_mask = Xarr.sum(axis=1) > 0
        non_zero_col_mask = Xarr.sum(axis=0) > 0

    # Subset AnnData (this keeps it backed where possible and avoids loading full dense matrix)
    adata_filtered = adata[non_zero_row_mask, non_zero_col_mask]
    print("Original shape:", adata.shape, "Filtered shape:", adata_filtered.shape)

    # Save filtered AnnData
    sc.write_h5ad(adata_filtered, "data/hematopoiesis_GSE129785_FACS_sorted_no_zeros.h5ad")