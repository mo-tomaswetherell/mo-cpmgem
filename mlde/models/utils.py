# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2024 Henry Addison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications to the original work have been made by Henry Addison
# to allow for conditional modelling.
"""All functions and modules related to model definition."""

from typing import Type

import numpy as np
import torch
from ml_collections.config_dict import ConfigDict

from .. import sde_lib


_MODELS: dict[str, Type[torch.nn.Module]] = {}
"""Mapping from model names to model classes."""


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config: Type[ConfigDict]) -> np.ndarray:
    """Get sigmas, i.e. the set of noise levels for SMLD from config files.

    Args:
        config: Configuration object.

    Returns:
        sigmas: A jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales
        )
    )

    return sigmas


def get_ddpm_params(config: ConfigDict) -> dict:
    """Get betas and alphas -- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000

    # Parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def create_model(config: ConfigDict, datadir: str) -> torch.nn.Module:
    """Create the score model.

    Args:
        config: Configuration object.
        datadir: Path to directory containing dataset folders.

    Returns:
        score_model: Score-based model.
    """
    model_name = config.model.name
    # TODO: This shouldn't really need the datadir. What it uses it for it to get the
    # variable names, which could be passed in themselves. Would need refactoring to do this.
    score_model = get_model(model_name)(config=config, datadir=datadir)
    score_model = score_model.to(config.device)
    # score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model: torch.nn.Module, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model (neural network).
        train: If `True`, the model is set to training mode. Otherwise, it is set to evaluation
            mode.

    Returns:
        A function that computes the output of the score-based model. The function takes in a
        mini-batch of training/evaluation data to model, conditioning inputs, and a mini-batch of
        conditioning variables for time steps.
    """

    def model_fn(x, cond, labels):
        """Compute the output of the score-based model.

        Sets the model to training or evaluation mode, and returns the output of the model.

        Args:
            x: A mini-batch of training/evaluation data to model.
            cond: A mini-batch of conditioning inputs.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted
                differently for different models.

        Returns:
            A tuple of (model output, new mutable states)  # TODO: Is this correct?
        """
        if not train:
            model.eval()
            return model(x, cond, labels)
        else:
            model.train()
            return model(x, cond, labels)

    return model_fn


def get_score_fn(
    sde: Type[sde_lib.SDE], model: torch.nn.Module, train: bool = False, continuous: bool = False
):
    """Return a score function that can be used for training and sampling.

    Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model (neural network).
        train: If True, the model is set to training mode. Otherwise, it is set to evaluation mode.
        continuous: If True, the score-based model is expected to directly take continuous time
            steps.

    Returns:
        score_fn: A function that computes and returns the output of the score-based model.

    Raises:
        NotImplementedError: If the SDE class is not supported.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x: torch.Tensor, cond: torch.Tensor, t) -> torch.Tensor:
            """Returns an estimate of the score.

            Args:
                x: A mini-batch of training/evaluation data to model.
                cond: A mini-batch of conditioning inputs.
                t:

            Returns:
                score: Estimated score.
            """
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, cond, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, cond, t):
            """Returns an estimate of the score.

            Args:
                x: A mini-batch of training/evaluation data to model.
                cond: A mini-batch of conditioning inputs.
                t:

            Returns:
                score: Estimated score
            """
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, cond, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x: torch.Tensor) -> np.ndarray:
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x: np.ndarray, shape: tuple) -> torch.Tensor:
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
