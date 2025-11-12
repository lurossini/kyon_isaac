import torch
import torch.nn as nn

from rsl_rl.utils import *

class CNN(nn.Module):
    def __init__(self,
                 resolution: tuple[int] | list[int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_channel_size: tuple[int] | list[int] = [8, 16],
                 kernel_size: tuple[int] | list[int] = [3, 3],
                 pool_kernel_size: tuple[int] | list[int] = [2, 2],
                 activation: str = "elu"):
        
        assert len(hidden_channel_size) != len(kernel_size), "In CNN module initialization: kernel_size and hidden_channel_size vectors have different lengths"
        super().__init__()
        self.layers = []

        # add first convolutional layer
        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_channel_size[0], kernel_size=kernel_size[0], padding=1))
        self.layers.append(resolve_nn_activation(activation))
        self.layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[0], stride=2))

        for layer_index in range(len(hidden_channel_size) - 1):
            self.layers.append(nn.Conv2d(in_channels=hidden_channel_size[layer_index], out_channels=hidden_channel_size[layer_index+1], kernel_size=kernel_size[layer_index+1], padding=1))
            self.layers.append(resolve_nn_activation(activation))
            self.layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[layer_index+1], stride=2))

        # add fully connected layer
        input_dim = (resolution[0] / len(pool_kernel_size)) * (resolution[1] / len(pool_kernel_size)) * hidden_channel_size[-1]
        self.layers.append(nn.Linear(in_features=input_dim, out_features=latent_dim)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
