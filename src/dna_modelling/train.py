import torch
import torch.nn as nn
from pathlib import Path

from visualize import plot_latent_embeddings
from visualize import plot_loss_curve
from visualize import plot_embeddings
from evaluate import *
from datetime import datetime

def sample_training_pairs(Aij, batch_size, positive_fraction=0.5, device="cpu"):
    """
    Samples positive and negative cell-peak pairs from Aij.

    Aij should be a torch tensor with shape:
        (n_cells, n_features)

    Returns:
        cell_idx, feature_idx, labels
    """

    n_pos = int(batch_size * positive_fraction)
    n_neg = batch_size - n_pos

    # Positive pairs where Aij == 1
    pos_cells, pos_features = torch.where(Aij > 0)

    pos_choice = torch.randint(
        low=0,
        high=pos_cells.shape[0],
        size=(n_pos,),
        device=Aij.device
    )

    batch_pos_cells = pos_cells[pos_choice]
    batch_pos_features = pos_features[pos_choice]
    batch_pos_labels = torch.ones(n_pos, device=Aij.device)

    # Negative pairs sampled from zeros
    n_cells, n_features = Aij.shape

    neg_cells_list = []
    neg_features_list = []

    while len(neg_cells_list) < n_neg:
        candidate_cells = torch.randint(
            0, n_cells, size=(n_neg,), device=Aij.device
        )
        candidate_features = torch.randint(
            0, n_features, size=(n_neg,), device=Aij.device
        )

        mask = Aij[candidate_cells, candidate_features] == 0

        neg_cells_list.append(candidate_cells[mask])
        neg_features_list.append(candidate_features[mask])

        current = sum(x.numel() for x in neg_cells_list)
        if current >= n_neg:
            break

    batch_neg_cells = torch.cat(neg_cells_list)[:n_neg]
    batch_neg_features = torch.cat(neg_features_list)[:n_neg]
    batch_neg_labels = torch.zeros(n_neg, device=Aij.device)

    cell_idx = torch.cat([batch_pos_cells, batch_neg_cells])
    feature_idx = torch.cat([batch_pos_features, batch_neg_features])
    labels = torch.cat([batch_pos_labels, batch_neg_labels])

    # Shuffle batch
    perm = torch.randperm(batch_size, device=Aij.device)

    return cell_idx[perm], feature_idx[perm], labels[perm]

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
    for epoch in range(model.epochs):
        optimizer.zero_grad()

        cell_idx, feature_idx, labels = sample_training_pairs(
            model.Aij,
            batch_size=65536,
            positive_fraction=0.5,
            device=model.device
        )

        logits = model(cell_idx, feature_idx)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 and epoch > 0:
            start_time = datetime.now()
            auc, _, f1_score, pr_auc, _ = validate_pairwise(model, targets, target_zeros)
            aucs.append(auc)
            f1_scores.append(f1_score)
            pr_aucs.append(pr_auc)
            print(f"Start: {start_time} | End: {datetime.now()} | Epoch {epoch}/{model.epochs} | Loss: {loss.item():.4f} | AUC (100%): {auc:.4f} | F1 Score: {f1_score:.4f} | PR AUC: {pr_auc:.4f}", flush=True)
            losses_per_interval.append(loss.item())
            interval_steps.append(epoch)

            with open(f"results/{ls_dim}_weighting_{model.weighting}_run{model.index}.csv", "a") as f:
                f.write(f"{loss.item()},{auc},{f1_score},{pr_auc}\n")

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