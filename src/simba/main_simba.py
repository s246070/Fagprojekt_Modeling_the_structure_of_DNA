import argparse
import random
from pathlib import Path
import contextlib
import sys

import numpy as np
import simba as si
import scipy.sparse as sp

from dna_modelling.utils.data import Data


def parse_args():
    parser = argparse.ArgumentParser(description="Run SIMBA on scATAC-seq data.")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--full", action="store_true", help="Use the full hematopoiesis dataset.")
    parser.add_argument("--run-name", type=str, default="simba_run", help="Name of this SIMBA run.")

    parser.add_argument("--min-cells", type=int, default=3, help="Minimum number of cells required for a peak.")
    parser.add_argument("--n-components", type=int, default=50, help="Number of PCA components for peak selection.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of PCs used for feature selection.")

    parser.add_argument("--embedding-dim", type=int, default=50, help="SIMBA/PBG embedding dimension.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU workers for SIMBA/PBG training.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of PBG training epochs.")

    parser.add_argument(
        "--skip-pc-feature-selection",
        action="store_true",
        help="Skip PCA-based peak selection and use all peaks after filtering.",
    )
    parser.add_argument(
        "--specify-path",
        type=str,
        default=None,
        help="Specify a custom path to an AnnData file for SIMBA input (overrides --full).",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Optional scratch/work directory for SIMBA/PBG intermediate files.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def make_dirs(run_name: str, workdir: str | None = None) -> tuple[Path, Path]:
    """Create SIMBA work/model directory and result directory."""

    if workdir is None:
        # Default: store SIMBA/PBG files in the project folder
        model_dir = Path("models") / "simba" / run_name
    else:
        # HPC mode: store SIMBA/PBG files on /work3 or another scratch location
        model_dir = Path(workdir) / run_name

    # Final useful outputs are still saved in the project folder
    result_dir = Path("results") / "simba" / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return model_dir, result_dir


def binarize_anndata(adata) -> None:
    """Binarize AnnData.X in-place without densifying sparse matrices."""
    if sp.issparse(adata.X):
        adata.X.data = (adata.X.data > 0).astype(np.float32)
        adata.X.eliminate_zeros()
    else:
        adata.X = (adata.X > 0).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_dir, result_dir = make_dirs(args.run_name, args.workdir)

    print(f"Starting SIMBA run: {args.run_name}", flush=True)
    print(f"SIMBA version: {si.__version__}", flush=True)
    print(f"Model directory: {model_dir}", flush=True)
    print(f"Result directory: {result_dir}", flush=True)

    # SIMBA writes pbg/, graph files, model files, and temporary outputs under workdir.
    si.settings.set_workdir(str(model_dir))

    data_loader = Data()

    # Important: backed=False because SIMBA preprocessing modifies the AnnData object.
    adata_cp = data_loader.load_data(backed=False, full=args.full, specify_path=args.specify_path)

    print(f"loaded data from path: {args.specify_path if args.specify_path else ('full dataset' if args.full else 'default dataset')}", flush=True)

    print("Loaded AnnData:", flush=True)
    print(adata_cp, flush=True)

    print("Checking AnnData index uniqueness...", flush=True)

    if not adata_cp.obs_names.is_unique:
        print("Warning: obs_names are not unique. Making cell names unique...", flush=True)
        adata_cp.obs_names_make_unique()

    if not adata_cp.var_names.is_unique:
        print("Warning: var_names are not unique. Making peak names unique...", flush=True)
        adata_cp.var_names_make_unique()

    print(f"obs_names unique: {adata_cp.obs_names.is_unique}", flush=True)
    print(f"var_names unique: {adata_cp.var_names.is_unique}", flush=True)
        
    # SIMBA scATAC tutorial uses an AnnData object called adata_CP:
    # rows = cells, columns = peaks.
    binarize_anndata(adata_cp)

    print("Filtering peaks...", flush=True)
    si.pp.filter_peaks(adata_cp, min_n_cells=args.min_cells)

    print("Calculating ATAC QC metrics...", flush=True)
    si.pp.cal_qc_atac(adata_cp)

    if args.skip_pc_feature_selection:
        use_top_pcs = False
        print("Skipping PCA-based peak selection. Graph will use all filtered peaks.", flush=True)
    else:
        use_top_pcs = True

        print("Running PCA for peak selection...", flush=True)
        si.pp.pca(adata_cp, n_components=args.n_components)

        print(f"Selecting top {args.n_pcs} PCs...", flush=True)
        si.pp.select_pcs(adata_cp, n_pcs=args.n_pcs)

        print("Selecting features associated with selected PCs...", flush=True)
        si.pp.select_pcs_features(adata_cp)

    preprocessed_path = result_dir / "adata_cp_preprocessed.h5ad"
    adata_cp.write_h5ad(preprocessed_path)
    print(f"Saved preprocessed AnnData to: {preprocessed_path}", flush=True)

    print("Generating SIMBA graph...", flush=True)
    si.tl.gen_graph(
        list_CP=[adata_cp],
        copy=False,
        use_top_pcs=use_top_pcs,
        dirname="graph0",
    )

    print("Configuring PBG training...", flush=True)
    pbg_params = si.settings.pbg_params.copy()
    pbg_params["dimension"] = args.embedding_dim
    pbg_params["workers"] = args.workers
    pbg_params["num_epochs"] = args.num_epochs

    print("Training SIMBA/PBG model...", flush=True)
    pbg_log_path = result_dir / "pbg_train.log"

    with open(pbg_log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file):
            si.tl.pbg_train(
                pbg_params=pbg_params,
                auto_wd=True,
                save_wd=True,
                output="model",
            )

    print("Reading embeddings...", flush=True)
    dict_adata = si.read_embedding(num_epochs=args.num_epochs)

    adata_cells = dict_adata["C"]
    adata_peaks = dict_adata["P"]

    # Add cell annotations back to the cell embedding object when possible.
    # The SIMBA tutorial does this with:
    # adata_C.obs["celltype"] = adata_CP[adata_C.obs_names, :].obs["celltype"].copy()
    for col in adata_cp.obs.columns:
        try:
            adata_cells.obs[col] = adata_cp[adata_cells.obs_names, :].obs[col].copy()
        except Exception:
            pass

    cell_path = result_dir / "adata_cells_simba.h5ad"
    peak_path = result_dir / "adata_peaks_simba.h5ad"

    adata_cells.write_h5ad(cell_path)
    adata_peaks.write_h5ad(peak_path)

    print(f"Saved cell embeddings to: {cell_path}", flush=True)
    print(f"Saved peak embeddings to: {peak_path}", flush=True)

    print("SIMBA run finished successfully.", flush=True)


if __name__ == "__main__":
    main()