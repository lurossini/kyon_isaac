import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from rsl_rl.utils import *

class CNN(nn.Sequential):
    def __init__(self,
                 resolution: tuple[int] | list[int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_channel_size: tuple[int] | list[int] = [8, 16],
                 kernel_size: tuple[int] | list[int] = [3, 3],
                 pool_kernel_size: tuple[int] | list[int] = [2, 2],
                 activation: str = "elu"):
        
        assert len(hidden_channel_size) == len(kernel_size), "In CNN module initialization: kernel_size and hidden_channel_size vectors have different lengths"
        super().__init__()
        layers = []

        # add first convolutional layer
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_channel_size[0], kernel_size=kernel_size[0], stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[0], stride=2))
        layers.append(resolve_nn_activation(activation))

        for layer_index in range(len(hidden_channel_size) - 1):
            layers.append(nn.Conv2d(in_channels=hidden_channel_size[layer_index], out_channels=hidden_channel_size[layer_index+1], kernel_size=kernel_size[layer_index+1], stride=1, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size[layer_index+1], stride=2))
            layers.append(resolve_nn_activation(activation))

        # add fully connected layer
        with torch.no_grad():
            dummy_image = torch.zeros(1, in_channels, *resolution)
            x = dummy_image
            for layer in layers:
                x = layer(x)
            image_feature_size = x.view(1, -1).shape[1]
        layers.append(nn.Linear(in_features=image_feature_size, out_features=latent_dim)) 
        layers.append(nn.LayerNorm(latent_dim))

        self._initialize_weights()

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def _initialize_weights(self):
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="tanh")
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        if isinstance(x, TensorDict):
            x = torch.cat([x["rgb"], x["distance_to_image_plane"]], dim=1)
        x = torch.permute(x, (0, 3, 1, 2))
        if x.dtype == torch.uint8:
            x = x.float()
        for layer in self:
            if isinstance(layer, nn.Linear):
                x = x.reshape(x.size(0), -1)
            x = layer(x)
        return x
