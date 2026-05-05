import torch
import torch.nn as nn
from pathlib import Path

from visualize import plot_latent_embeddings
from visualize import plot_loss_curve
from visualize import plot_embeddings
from evaluate import *

def TrainModel(model, device="cpu", plots=False, targets=None, target_zeros=None):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    # Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
    # n_pos = model.Aij.sum()
    # n_neg = model.Aij.numel() - n_pos
    # pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Calculate BCELoss with logits without pos_weight
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    losses_per_interval = []
    interval_steps = []
    ls_dim = model.ls_dim
    print(f"Starting training for {model.epochs} epochs with learning rate {model.lr} and latent space dimension {ls_dim}...", flush=True)
    for epoch in range(model.epochs):
        optimizer.zero_grad()

        logits = model()
        loss = criterion(logits, model.Aij)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 50 == 0 and epoch > 0:
            auc, _, f1_score, pr_auc, _ = validate(model, model.Aij, targets, target_zeros)
            print(f"Epoch {epoch}/{model.epochs} | Loss: {loss.item():.4f} | AUC (100%): {auc:.4f} | F1 Score: {f1_score:.4f} | PR AUC: {pr_auc:.4f}", flush=True)
            losses_per_interval.append(loss.item())
            interval_steps.append(epoch)

            if epoch % 200 == 0 and plots:
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
                        interval_steps=interval_steps,
                        losses_per_interval=losses_per_interval,
                    )
                else:
                    plot_embeddings(
                        model=model,
                        ls_dim=ls_dim,
                        epoch=epoch,
                        interval_steps=interval_steps,
                        losses_per_interval=losses_per_interval,
                    )

    return losses