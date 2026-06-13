import argparse
import torch

from dna_modelling.utils import Data
from dna_modelling.model import LDM
from dna_modelling.train import TrainModel
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train the DNA latent distance model on a prepared train set.")
    parser.add_argument("--seed", type=int, default=1, help="Seed used to pick the prepared train-set file.")
    parser.add_argument("--ls-dim", type=int, default=2, help="Latent space dimension.")
    parser.add_argument("--index", type=int, default=1, help="Run index used for saved artifacts.")
    parser.add_argument(
        "--weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable positive-class weighting in the loss.",
    )
    parser.add_argument("--epochs", type=int, default=1001, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train in row batches instead of using the full matrix each step.",
    )
    parser.add_argument("--num-blocks", type=int, default=100, help="Number of row blocks per epoch.")
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

    # Load Aij, targets, and target_zeros from data/train_sets
    train_set_path = f"data/train_sets/Aij_train_{args.seed}.h5ad"
    Aij, targets, target_zeros = Data().load_test_set(train_set_path, device=device)

    print(f"converted data to tensor{datetime.now()}", flush=True)

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

    # Train
    losses = TrainModel(
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


if __name__ == "__main__":
    main()
