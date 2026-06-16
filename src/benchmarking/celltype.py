import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dna_modelling.utils import Data  # noqa: E402

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=True)

cell_types = adata.obs

with open("src/benchmarking/cell_types.txt", "w") as f:
    for i in range(len(cell_types)):
        f.write(f"{cell_types['cell_type'].iloc[i]}\n")

# Load AnnData
data_loader = Data()
adata = data_loader.load_data(backed=True, full=False, specify_path="train_sets/adata_subset_10k_1.h5ad")

cell_types = adata.obs
print(cell_types.shape)

with open("src/benchmarking/cell_types_subset_1.txt", "w") as f:
    for i in range(len(cell_types)):
        f.write(f"{cell_types['cell_type'].iloc[i]}\n")