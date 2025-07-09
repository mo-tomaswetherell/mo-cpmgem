"""Module for generating samples using a trained score-based model."""

import itertools
import logging
from pathlib import Path
from typing import Type

import torch
import xarray as xr
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ml_collections.config_dict import ConfigDict

import mlde.models as models
import mlde.sampling as sampling
from mlde.data import np_samples_to_xr
from mlde.utils.training.dataset import get_variables
from mlde.utils.transforms import ComposeT
from mlde.losses import get_optimizer
from mlde.models.ema import (
    ExponentialMovingAverage,
)
from mlde.models.location_params import (
    LocationParams,
)
from mlde.utils import restore_checkpoint
from mlde.models import utils as mutils, cncsnpp, cunet, layerspp, layers, normalization
from mlde.sde_lib import (
    VESDE,
    VPSDE,
    subVPSDE,
)

logger = logging.getLogger(__name__)


def _init_state(config: ConfigDict, datadir: str) -> dict:
    """Initialise model, optimiser and exponential moving average.

    Args:
        config: Configuration object.
        datadir: Path to directory containing dataset folders.

    Returns:
        state: Dictionary containing the model, optimiser, location parameters, and exponential
            moving average.
    """
    score_model = mutils.create_model(config, datadir)

    location_params = LocationParams(config.model.loc_spec_channels, config.data.image_size)
    location_params = location_params.to(config.device)
    # location_params = torch.nn.DataParallel(location_params)

    optimizer = get_optimizer(
        config, itertools.chain(score_model.parameters(), location_params.parameters())
    )
    ema = ExponentialMovingAverage(
        itertools.chain(score_model.parameters(), location_params.parameters()),
        decay=config.model.ema_rate,
    )

    state = dict(
        step=0,
        optimizer=optimizer,
        model=score_model,
        location_params=location_params,
        ema=ema,
    )

    return state


def load_model(
    config: ConfigDict, ckpt_filename: str | Path, datadir: str | Path
) -> tuple[dict, callable, list[str]]:
    """Load a trained score-based model and a sampling function.

    Args:
        config: Configuration object.
        ckpt_filename: Path to the model checkpoint file.
        datadir: Path to directory containing dataset folders.

    Returns:
        state: Dictionary containing the model, optimiser, location parameters, and exponential
            moving average.
        sampling_fn: Sampling function. Callable that generates samples using the trained
            score-based model.
        target_vars: List of target variable names, e.g. ["target_pr"].

    Raises:
        RuntimeError: If the SDE specified in the config is not one of 'vesde', 'vpsde', or
            'subvpsde'. Also raised if no file is found at the path specified by `ckpt_filename`.
    """
    deterministic = "deterministic" in config and config.deterministic

    # Set up the SDE object.
    if deterministic:
        sde = None
        sampling_eps = 0
    else:
        if config.training.sde == "vesde":
            sde = VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=config.model.num_scales,
            )
            sampling_eps = 1e-5
        elif config.training.sde == "vpsde":
            sde = VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
            sampling_eps = 1e-3
        elif config.training.sde == "subvpsde":
            sde = subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
            sampling_eps = 1e-3
        else:
            raise RuntimeError(f"Unknown SDE {config.training.sde}")

    # Load model from checkpoint
    state = _init_state(config, datadir)
    state, loaded = restore_checkpoint(ckpt_filename, state, config.device)
    if not loaded:
        raise RuntimeError(f"No checkpoint file found at {ckpt_filename}")

    state["ema"].copy_to(state["model"].parameters())

    # Load the sampling function
    _, target_vars = get_variables(datadir, config.data.dataset_name)
    num_output_channels = len(target_vars)
    sampling_shape = (
        config.eval.batch_size,
        num_output_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    return state, sampling_fn, target_vars


def generate_np_sample_batch(
    sampling_fn: callable,
    score_model: torch.nn.Module,
    config: ConfigDict,
    cond_batch: torch.Tensor,
) -> np.ndarray:
    """Use the score-based model and conditioning data to generate a batch of samples.

    Uses the sampling function to generate samples using the score model, conditioned on the
    conditioning data. The samples are then moved to the CPU and converted to numpy.

    Args:
        sampling_fn: Sampling function.
        score_model: Score-based model (neural network).
        config: Configuration object.
        cond_batch: Batch of data to condition the sample generation on.

    Returns:
        samples: Array of generated samples.
    """
    cond_batch = cond_batch.to(config.device)

    # Generate samples using the score-based model and conditioning data
    samples = sampling_fn(score_model, cond_batch)[0]

    # Move samples to CPU and convert to numpy
    samples = samples.cpu().numpy()

    return samples


def sample(
    sampling_fn: callable,
    state: dict,
    config: ConfigDict,
    eval_dl: torch.utils.data.DataLoader,
    target_transform: Type[ComposeT],
    target_vars: list[str],
) -> xr.Dataset:
    """Generate samples for the entire dataset.

    Args:
        sampling_fn: Sampling function. Callable that generates samples using the score-based model
            and conditioning data.
        state: Dictionary containing the model, optimiser, location parameters, and exponential
            moving average.
        config: Configuration object.
        eval_dl: Dataloader to load the conditioning data from.
        target_transform: Transform object representing the target transform pipeline.
        target_vars: List of target variable names, e.g. ["target_pr"].

    Returns:
        ds: Dataset containing the generated samples for the entire dataset.
    """
    xr_sample_batches: list[xr.Dataset] = []
    score_model = state["model"]
    location_params = state["location_params"]

    cf_data_vars = {
        key: eval_dl.dataset.ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
    }

    with logging_redirect_tqdm():
        with tqdm(total=len(eval_dl.dataset), desc=f"Sampling", unit=" timesteps") as pbar:
            for cond_batch, _, time_batch in eval_dl:
                logger.info("Loaded batch of conditioning data")
                # Append any location-specific parameters
                cond_batch = location_params(cond_batch)

                coords = eval_dl.dataset.ds.sel(time=time_batch).coords

                logger.info("Starting sample generation")
                np_sample_batch = generate_np_sample_batch(
                    sampling_fn, score_model, config, cond_batch
                )
                logger.info("Finished sample generation")

                xr_sample_batch = np_samples_to_xr(
                    np_sample_batch,
                    target_transform,
                    target_vars,
                    coords,
                    cf_data_vars,
                )
                logger.info("Converted samples to xarray")

                xr_sample_batches.append(xr_sample_batch)
                pbar.update(cond_batch.shape[0])

    # Combine all the batches of samples into a single dataset
    ds = xr.combine_by_coords(
        xr_sample_batches,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )

    return ds
