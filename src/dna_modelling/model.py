import torch
import torch.nn as nn

class LDM(nn.Module):
    """
    Binary latent distance model for cell x peak accessibility matrices.

    Args:
        data (torch.Tensor): Input binary matrix of shape (cells, features).
        ls_dim (int): Dimensionality of the latent space.
        device (torch.device): Device to run the model on.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        seed (int, optional): Random seed for reproducibility. Default is None.
    
    Attributes:
        Aij (torch.Tensor): Input binary matrix.
        ls_dim (int): Dimensionality of the latent space.
        device (torch.device): Device to run the model on.
        cells (int): Number of cells in the input matrix.
        features (int): Number of features in the input matrix.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        seed (int, optional): Random seed for reproducibility.
        embed_cells (torch.nn.Parameter): Latent embeddings for cells.
        embed_features (torch.nn.Parameter): Latent embeddings for features.
        cell_bias (torch.nn.Parameter): Bias terms for cells.
        feature_bias (torch.nn.Parameter): Bias terms for features.

    returns:
        torch.Tensor: Logits representing the predicted accessibility probabilities.

    def __init__:
        Initializes the LDM model with the given parameters and sets up the latent embeddings and bias terms.

    def __set_seed:
        Sets the random seed for reproducibility.

    def forward:
        Computes the logits for the predicted accessibility probabilities based on the latent embeddings and bias terms.

    def probabilities:
        Computes the predicted accessibility probabilities by applying the sigmoid function to the logits.
    
    """
    def __init__(self, data, ls_dim, device, epochs, lr, index, weighting, seed=None):
        super().__init__()
        self.Aij = data.float().to(device)
        self.ls_dim = ls_dim
        self.device = device
        self.cells, self.features = self.Aij.shape
        self.epochs = epochs
        self.lr = lr
        self.index = index
        self.weighting = weighting
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
    
    def __set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def forward(self, rows=None):
        if rows is None:
            # Full matrix: [n_cells, n_features]
            embed_cells = self.embed_cells
            cell_bias = self.cell_bias
        else:
            # Batched rows: [batch_size, n_features]
            embed_cells = self.embed_cells[rows]
            cell_bias = self.cell_bias[rows]
        
        dist = torch.cdist(embed_cells, self.embed_features, p=2)

        logits = (cell_bias[:, None] + self.feature_bias[None, :] - dist)

        return logits

    def probabilities(self):
        return torch.sigmoid(self.forward())

    def save_model(self, path):
        torch.save(self.state_dict(), path)