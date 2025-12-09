import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from rsl_rl.utils import *

class CNN(nn.Module):
    def __init__(self,
                 resolution: tuple[int, int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_channel_size: tuple[int] | list[int] = (8, 16),
                 kernel_size: tuple[int] | list[int] = (3, 3),
                 stride: tuple[int] | list[int] = (2, 2),
                 activation: str = "elu"):
        super().__init__()
        act = resolve_nn_activation(activation)

        # conv stack
        convs = []
        convs.append(nn.Conv2d(in_channels, hidden_channel_size[0],
                               kernel_size=kernel_size[0], stride=stride[0], padding=1))
        convs.append(act)

        for i in range(len(hidden_channel_size) - 1):
            convs.append(nn.Conv2d(hidden_channel_size[i],
                                   hidden_channel_size[i + 1],
                                   kernel_size=kernel_size[i + 1],
                                   stride=stride[i + 1], padding=1))
            convs.append(act)

        self.conv = nn.Sequential(*convs)

        # compute flattened size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, resolution[0], resolution[1])
            h = self.conv(dummy)
            flat_dim = h.view(1, -1).shape[1]

        # MLP head
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 1024), act,
            nn.Linear(1024, 256), act,
            nn.Linear(256, 64), act,
            nn.Linear(64, latent_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        # accept TensorDict-like input as your prior code
        if isinstance(x, TensorDict):
            x = torch.cat([x["rgb"], x["distance_to_image_plane"]], dim=-1)

        # x expected shape: (B, H, W, C) -> convert to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float()
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        # conv init
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # fc init - use Xavier for linear layers (works well with many activations)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)