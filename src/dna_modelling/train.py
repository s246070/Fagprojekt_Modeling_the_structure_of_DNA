import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from dna_modelling.utils.visualize import plot_latent_embeddings
from dna_modelling.utils.visualize import plot_loss_curve
from dna_modelling.utils.visualize import plot_embeddings
from dna_modelling.evaluate import validate


def TrainModel(
    model,
    device="cpu",
    plots=False,
    targets=None,
    target_zeros=None,
    batching=False,
    num_blocks=10,
    validation=True,
):
    """
    Train the DNA sequence embedding model.

    Args:
        model: The DNA sequence embedding model to be trained.
        device: The device to run the training on (e.g., 'cpu' or 'cuda').
        plots: Whether to save visualization plots during training.
    Attributes:
        optimizer: The optimizer used for training the model.
        criterion: The loss function used for training the model.
        losses: A list to store the loss values for each epoch.
        losses_per_interval: A list to store the loss values at specified intervals.
        interval_steps: A list to store the epoch numbers at specified intervals.
        ls_dim: The dimension of the latent space used in the model.
        
    Returns:
        A list of loss values for each epoch during training.

    """
    model.to(device)
    model.train()
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    # Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance or use standard BCE loss if weighting is not enabled
    if model.weighting:
        n_pos = model.Aij.sum()
        n_neg = model.Aij.numel() - n_pos
        pos_weight = (n_neg / (n_pos + 1e-8)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()


    losses = []
    aucs = []
    pr_aucs = []
    f1_scores = []
    losses_per_interval = []
    interval_steps = []
    ls_dim = model.ls_dim
    print(f"Starting training for {model.epochs} epochs with learning rate {model.lr} and latent space dimension {ls_dim}...", flush=True)

    def handle_checkpoint(epoch, current_loss):
        if epoch % 100 != 0 and epoch < 0:
            return

        if validation:
            start_time = datetime.now()
            auc, _, f1_score, pr_auc, _ = validate(model, targets, target_zeros)
            aucs.append(auc)
            f1_scores.append(f1_score)
            pr_aucs.append(pr_auc)

            print(
                f"Start: {start_time} | End: {datetime.now()} | "
                f"Epoch {epoch}/{model.epochs} | "
                f"Loss: {current_loss:.4f} | "
                f"AUC (100%): {auc:.4f} | "
                f"F1 Score: {f1_score:.4f} | "
                f"PR AUC: {pr_auc:.4f}",
                flush=True,
            )

            with open(
                f"results/ldm_ls{ls_dim}_blocks{num_blocks}_index{model.index}.csv",
                "a",
            ) as f:
                f.write(f"{current_loss},{auc},{f1_score},{pr_auc}\n")

        losses_per_interval.append(current_loss)
        interval_steps.append(epoch)
        print(f"Epoch {epoch}/{model.epochs} | time: {datetime.now()}", flush=True)
        model.save_model(f"models/ldm_ls{ls_dim}_epoch{epoch}_blocks{num_blocks}_index{model.index}.pth")

        if plots:
            plot_loss_curve(
                ls_dim=ls_dim,
                interval_steps=interval_steps,
                losses_per_interval=losses_per_interval,
            )

            if model.ls_dim > 2:
                plot_latent_embeddings(
                    model=model,
                    ls_dim=ls_dim,
                    epoch=epoch,
                )
            else:
                plot_embeddings(
                    model=model,
                    ls_dim=ls_dim,
                    epoch=epoch,
                    interval_steps=interval_steps,
                    losses_per_interval=losses_per_interval,
                )
    
    if batching:
        n_nodes = model.Aij.shape[0]

        for epoch in range(model.epochs):
            epoch_loss = 0.0

            # Shuffle row indices each epoch
            perm = torch.randperm(n_nodes, device=device)

            # Split rows into blocks
            blocks = torch.chunk(perm, num_blocks)

            for batch_idx in blocks:
                optimizer.zero_grad()

                # Forward only selected rows
                logits = model(rows=batch_idx)

                # Target block: same rows from Aij
                target_batch = model.Aij[batch_idx, :].to(device)

                loss = criterion(logits, target_batch)

                loss.backward()
                optimizer.step()

                # Weighted average loss over all row-blocks
                epoch_loss += loss.item() * (len(batch_idx) / n_nodes)

            losses.append(epoch_loss)

            handle_checkpoint(epoch, epoch_loss)

        return losses

    else:
        for epoch in range(model.epochs):
            optimizer.zero_grad()

            logits = model()

            loss = criterion(logits, model.Aij)

            loss.backward()

            optimizer.step()

            handle_checkpoint(epoch, loss.item())

        return losses