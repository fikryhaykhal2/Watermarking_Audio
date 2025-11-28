import torch
import torch.nn as nn
import numpy as np

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        pass 

    def forward(self, x):
        return x

def apply_cnn_denoising(y_wm: np.ndarray, sr: int, model: CNNAutoencoder) -> np.ndarray:
    return y_wm