import argparse
import torch

from dna_modelling.utils import Data
from dna_modelling.model import LDM
from dna_modelling.train import TrainModel
from dna_modelling.evaluate import make_test_set, validate
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DNA latent distance model.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--ls-dim", type=int, default=2, help="Latent space dimension.")
    parser.add_argument("--index", type=int, default=1, help="Run index used for saved artifacts.")
    parser.add_argument(
        "--full-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the full AnnData dataset instead of the smaller default dataset.",
    )
    parser.add_argument("--epochs", type=int, default=1001, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable positive-class weighting in the loss.",
    )
    parser.add_argument(
        "--batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train in row batches instead of using the full matrix each step.",
    )
    parser.add_argument("--num-blocks", type=int, default=1000, help="Number of row blocks per epoch.")
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation during training checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"everything is imported!{datetime.now()}, device: {device}", flush=True)

    # Load AnnData
    data_loader = Data()
    adata = data_loader.load_data(backed=True, full=args.full_data)

    print(f"data added{datetime.now()}", flush=True)

    # Convert AnnData.X to tensor
    Aij = data_loader.anndata_to_tensor(adata, device=device, make_binary=True)

    print(f"converted data to tensor{datetime.now()}", flush=True)

    Aij, targets, target_zeros = make_test_set(Aij, percentage=0.1)

    print(f"created test set{datetime.now()}", flush=True)

    # Initialize model
    model = LDM(
        data=Aij,
        ls_dim=args.ls_dim,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weighting=args.weighting,
        index=args.index,
        seed=args.seed,
    )

    print(f"beginning training{datetime.now()}", flush=True)

    auc, auc_data, f1_score, pr_auc, pr_curve_data = validate(model, targets, target_zeros)

    print(f"AUROC: {auc:.4f}", flush=True)
    print(f"F1 Score: {f1_score:.4f}", flush=True)
    print(f"PR AUC: {pr_auc:.4f}", flush=True)

    # Train
    TrainModel(
        model,
        device=device,
        plots=False,
        targets=targets,
        target_zeros=target_zeros,
        batching=args.batching,
        num_blocks=args.num_blocks,
        validation=args.validation,
    )

    print(f"training complete{datetime.now()}", flush=True)

    auc, auc_data, f1_score, pr_auc, pr_curve_data = validate(model, targets, target_zeros)

    print(f"AUROC: {auc:.4f}", flush=True)
    print(f"F1 Score: {f1_score:.4f}", flush=True)
    print(f"PR AUC: {pr_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()