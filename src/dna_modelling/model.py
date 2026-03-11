from torch import nn
import torch
from torch.distributions import sigmoid


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
        

    def __set_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def forward(self):
        """Forward pass to reconstruct the data matrix."""
        # Compute pairwise distances between cell and feature embeddings
        dist = torch.cdist(self.embed_cells, self.embed_features)
        # Reconstruct the data matrix 
        recon = self.gamma * dist + self.beta
        return recon
    
    from torch.distributions import Normal

    def probit(self):
        """
        Compute binary edge probabilities using a probit latent distance model.

        Returns:
            prob_matrix: [D x E] matrix of edge probabilities
            latent_var: [D x E] matrix of latent scores
        """


        # Linear term βᵀx_ij
        linear_term = torch.matmul(self.Aij, self.beta.unsqueeze(1))

        # Distance term -||w_i - v_j||
        dist = -torch.norm(self.w.unsqueeze(1) - self.v.unsqueeze(0), dim=2)

        # Latent score
        latent_var = self.gamma + linear_term + dist

        # Probit probability
        prob_matrix = torch.sigmoid(latent_var)

        return prob_matrix, latent_var
