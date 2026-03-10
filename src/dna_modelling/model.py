from torch import nn
import torch


class LDM(nn.Module):
    """"
    Latent Distance Model (LDM) for DNA modelling. This model learns latent representations for cells and features, 
    and uses them to reconstruct the original data matrix. 
    The model is trained using mean squared error loss between the original data and the reconstructed data.
    """
    def __init__(self, data, ls_dim, device, epochs, lr, seed=None):
        super(LDM, self).__init__()
        self.data = data
        self.ls_dim = ls_dim
        self.device = device

        # data dimensions
        self.cells, self.features = data.shape

        # Hyperparameters
        self.epochs = epochs
        self.lr = lr

        # Set random seed for reproducibility
        self.seed = seed
        self.__set_seed(seed)

        # Learned parameters
        self.gamma = nn.Parameter(torch.randn(1, device=device))
        self.beta = nn.Parameter(torch.randn(1, device=device))
        self.embed_cells = torch.nn.Parameter(torch.randn(self.cells, self.ls_dim, device=device))
        self.embed_features = torch.nn.Parameter(torch.randn(self.features, self.ls_dim, device=device))

        