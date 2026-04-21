import torch
import torch.nn as nn

class LDM(nn.Module):
    """
    Binary latent distance model for cell x peak accessibility matrices.
    """
    def __init__(self, data, ls_dim, device, epochs, lr, seed=None):
        super().__init__()
        self.Aij = data.float().to(device)
        self.ls_dim = ls_dim
        self.device = device

        self.cells, self.features = self.Aij.shape
        self.epochs = epochs
        self.lr = lr

        self.seed = seed
        self.__set_seed(seed)

        # Latent embeddings
        self.embed_cells = nn.Parameter(
            torch.randn(self.cells, self.ls_dim, device=device) * 0.01
        )
        self.embed_features = nn.Parameter(
            torch.randn(self.features, self.ls_dim, device=device) * 0.01
        )

        # Bias terms
        self.cell_bias = nn.Parameter(torch.zeros(self.cells, device=device))
        self.feature_bias = nn.Parameter(torch.zeros(self.features, device=device))

        print("Class initialized")
    def __set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def forward(self):
        dist = torch.cdist(self.embed_cells, self.embed_features, p=2)
        logits = (
            self.cell_bias[:, None]
            + self.feature_bias[None, :]
            - dist
        )
        return logits

    def probabilities(self):
        return torch.sigmoid(self.forward())

    def save_model(self, path):
        torch.save(self.state_dict(), path)