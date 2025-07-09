# TODO: The unet module does not exist. This module should therefore not be used, and may be
# removed.
import logging

import torch.nn as nn
from ml_collections.config_dict import ConfigDict

from src.mlde.unet import unet  # Will throw an error
from . import utils


def create_model(config, num_predictors):
    if config.model.name == "u-net":
        return unet.UNet(num_predictors, 1)


from src.mlde_utils.training.dataset import get_variables

######################################
# !!!! DETERMINISTIC ONLY       !!!! #
# This model does not use the time   #
# or denoising channels at all       #
######################################


@utils.register_model(name="det_cunet")
class DetPredNet(nn.Module):
    """A purely deterministic plain U-Net with conditioning input."""

    def __init__(self, config: ConfigDict, datadir: str):
        """Initialize a deterministic U-Net.

        Args:
            config: Configuration object.
            datadir: Path to directory containing dataset folders.
        """
        if not config.deterministic:
            logging.warning("Only use det_cunet for deterministic approach")

        super().__init__()
        self.config = config

        cond_var_channels, output_channels = list(
            map(len, get_variables(datadir, config.data.dataset_name))
        )
        if config.data.time_inputs:
            cond_time_channels = 3
        else:
            cond_time_channels = 0
        input_channels = cond_var_channels + cond_time_channels + config.model.loc_spec_channels

        self.unet = unet.UNet(input_channels, output_channels)

    def forward(self, x, cond, t):
        """Forward of conditioning inputs through the deterministic U-Net model.

        Since not using the score-based, denoising approached, do not need to pass the time or the channels to be denoised to the model."""
        return self.unet(cond)
