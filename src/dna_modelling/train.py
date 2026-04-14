import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import umap
import numpy as np
from pathlib import Path

def TrainModel(model, device="cpu", threads=1):
    torch.set_num_threads(threads)
    model.to(device)
    model.train()
    Path("plots").mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    # Class imbalance handling
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

        if epoch %100 == 0:
            print(f"Epoch {epoch}/{model.epochs} | Loss: {loss.item():.4f}")
            losses_per_interval.append(loss.item())
            interval_steps.append(epoch)


        if epoch %1_000 == 0 and epoch != 0:
            
            # 1. Initialize the UMAP model
            # n_neighbors: Controls local vs global structure (usually 5 to 50)
            # min_dist: Controls how tightly points are packed (0.0 to 0.99)

            reducer = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean'
                    )

            # torch.save(model.state_dict(),f'models/model_{epoch}')

            # 2. Fit and transform the embeddings
            cell_embeddings = model.embed_cells.detach().cpu().numpy()
            peak_embeddings = model.embed_features.detach().cpu().numpy()
            stacked_embeddings = np.vstack([cell_embeddings, peak_embeddings])
            stacked_embeddings_2d = reducer.fit_transform(stacked_embeddings)

            n_cells = cell_embeddings.shape[0]
            cell_embeddings_2d = stacked_embeddings_2d[:n_cells]
            peak_embeddings_2d = stacked_embeddings_2d[n_cells:]

            # 3. visualize the 2D embeddings
            plt.figure(figsize=(7, 6))

            plt.scatter(cell_embeddings_2d[:, 0], cell_embeddings_2d[:, 1],
                        s=5, alpha=0.7, color="blue", label="Cells")

            plt.scatter(peak_embeddings_2d[:, 0], peak_embeddings_2d[:, 1],
                        s=5, alpha=0.7, color="red", label="Peaks")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title(f"Cell & Peak latent space (ls_dim={ls_dim})")
            plt.legend()

            plt.savefig(f"plots/latent_space_ls{ls_dim}_{epoch}.png", dpi=300)
            plt.close()

            plt.figure(figsize=(7, 6))

            plt.scatter(cell_embeddings_2d[:, 0], cell_embeddings_2d[:, 1],
                        s=5, alpha=0.7, color="blue", label="Cells")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title(f"Cell latent space (ls_dim={ls_dim})")
            plt.legend()

            plt.savefig(f"plots/latent_space_cells_ls{ls_dim}_{epoch}.png", dpi=300)
            plt.close()

            plt.figure(figsize=(7, 6))

            plt.scatter(peak_embeddings_2d[:, 0], peak_embeddings_2d[:, 1],
                        s=5, alpha=0.7, color="red", label="Peaks")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title(f"Peak latent space (ls_dim={ls_dim})")
            plt.legend()

            plt.savefig(f"plots/latent_space_peaks_ls{ls_dim}_{epoch}.png", dpi=300)
            plt.close()


            plt.figure(figsize=(7,6))
            plt.plot(interval_steps, losses_per_interval)
            plt.xlabel("Training step")
            plt.ylabel("Loss")
            plt.title(f"Training Loss (ls_dim={ls_dim})")
            plt.grid(True)

            plt.savefig(f"plots/loss_curve_ls{ls_dim}.png", dpi=300)
            plt.close()
    return losses