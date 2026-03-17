import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def TrainModel(model, device="cpu", threads=1):
    torch.set_num_threads(threads)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    # Class imbalance handling
    n_pos = model.Aij.sum()
    n_neg = model.Aij.numel() - n_pos
    pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = []
    losses_per_interval = []
    for epoch in range(model.epochs):
        optimizer.zero_grad()

        logits = model()
        loss = criterion(logits, model.Aij)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        if epoch %10_000 == 0:
            print(f"Epoch {epoch}/{model.epochs} | Loss: {loss.item():.4f}")
            losses_per_interval.append(loss.item())


        if epoch %100_000 == 0 and epoch != 0:
            torch.save(model.state_dict(),f'models/model_{epoch}')

            cell_embeddings = model.embed_cells.detach().cpu().numpy()
            peak_embeddings = model.embed_features.detach().cpu().numpy()
            plt.figure(figsize=(7, 6))

            plt.scatter(cell_embeddings[:, 0], cell_embeddings[:, 1],
                        s=5, alpha=0.7, color="blue", label="Cells")

            plt.scatter(peak_embeddings[:, 0], peak_embeddings[:, 1],
                        s=5, alpha=0.7, color="red", label="Peaks")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title("Cell & Peak latent space")
            plt.legend()

            plt.savefig(f"plots/latent_space{epoch}.png", dpi=300)

            plt.figure(figsize=(7, 6))

            plt.scatter(cell_embeddings[:, 0], cell_embeddings[:, 1],
                        s=5, alpha=0.7, color="blue", label="Cells")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title("Cell latent space")
            plt.legend()

            plt.savefig(f"plots/latent_space_cells_{epoch}.png", dpi=300)

            plt.figure(figsize=(7, 6))

            plt.scatter(peak_embeddings[:, 0], peak_embeddings[:, 1],
                        s=5, alpha=0.7, color="red", label="Peaks")

            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.title("Peak latent space")
            plt.legend()

            plt.savefig(f"plots/latent_space_peaks_{epoch}.png", dpi=300)


            plt.figure(figsize=(7,6))
            plt.plot(losses_per_interval)
            plt.xlabel("Training step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)

            plt.savefig("plots/loss_curve.png", dpi=300)
            plt.close()
    return losses