import torch
import torch.nn as nn
from pathlib import Path

from visualize import plot_latent_embeddings
from visualize import plot_loss_curve


def TrainModel(model, device="cpu", threads=1):
    """
    Train the DNA sequence embedding model.

    Args:
        model: The DNA sequence embedding model to be trained.
        device: The device to run the training on (e.g., 'cpu' or 'cuda').
        threads: The number of CPU threads to use for training.

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
    torch.set_num_threads(threads)
    model.to(device)
    model.train()
    Path("plots").mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    n_pos = model.Aij.sum()
    n_neg = model.Aij.numel() - n_pos
    pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = []
    losses_per_interval = []
    interval_steps = []
    ls_dim = model.ls_dim

    for epoch in range(model.epochs):
        optimizer.zero_grad()

        logits = model()
        loss = criterion(logits, model.Aij)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{model.epochs} | Loss: {loss.item():.4f}")
            losses_per_interval.append(loss.item())
            interval_steps.append(epoch)

        if epoch % 1_000 == 0 and epoch != 0:
            plot_latent_embeddings(
                model=model,
                ls_dim=ls_dim,
                epoch=epoch,
                interval_steps=interval_steps,
                losses_per_interval=losses_per_interval,
            )
            plot_loss_curve(
                ls_dim=ls_dim,
                interval_steps=interval_steps,
                losses_per_interval=losses_per_interval,
            )

    return losses