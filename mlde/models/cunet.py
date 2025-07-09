import functools
import logging

import numpy as np
import torch
import torch.nn as nn
from ml_collections.config_dict import ConfigDict

from src.mlde_utils.training.dataset import get_variables

from . import layers, layerspp, normalization, utils

#####################################
# !!!! MODEL ONLY FOR DEBUGGING!!!! #
#####################################

USABLE_IMAGE_SIZE = 28  # u-net architechture currently designed to work with 28x28 images


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


@utils.register_model(name="cunet")
class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture with conditioning input."""

    def __init__(self, config: ConfigDict, datadir: str):
        """Initialize a time-dependent score-based network.

        Args:
            config: Configuration object.
            datadir: Path to directory containing dataset folders.
        """
        logging.warning("Only use cunet for debugging")
        super().__init__()
        self.config = config
        # TODO: marginal_prob_std should be a function that takes time t and gives the standard
        # deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        marginal_prob_std = None
        cond_var_channels, output_channels = list(
            map(len, get_variables(datadir, config.data.dataset_name))
        )
        cond_time_channels = 3
        input_channels = (
            output_channels
            + cond_var_channels
            + cond_time_channels
            + config.model.loc_spec_channels
        )
        channels = [32, 64, 128, 256]  # The number of channels for feature maps of each resolution
        embed_dim = 256  # The dimensionality of Gaussian random feature embeddings

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], output_channels, 3, stride=1)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, cond, t):
        output_shape = x.shape
        # combine the modelled data and the conditioning inputs
        x = torch.cat([x, cond], dim=1)[..., :USABLE_IMAGE_SIZE, :USABLE_IMAGE_SIZE]
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)[..., :USABLE_IMAGE_SIZE, :USABLE_IMAGE_SIZE]
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = torch.nn.functional.interpolate(
            h, size=output_shape[-2:], mode="bilinear", align_corners=True
        )

        # TODO: Do I need to normalize with the marginal_prob_std? And what is it in this more complicated world? What is t in this framework?
        # Normalize output
        # h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
