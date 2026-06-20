import torch
import matplotlib.pyplot as plt

model = torch.load("models/ldm_ls2_weighting_False_run2.pth")
data = model['embed_cells'].cpu().detach().numpy()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
plt.tight_layout()
plt.savefig("plots/latent_space_small_visualization.png", dpi=300)
plt.show()