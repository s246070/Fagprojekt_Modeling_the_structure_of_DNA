import torch
import matplotlib.pyplot as plt
import numpy as np

name = "pca_full_variance_ratio"

data = np.load(f"models/{name}.npy")

print(data.shape, data)