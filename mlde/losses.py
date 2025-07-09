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
# to allow for sampling
"""All functions related to loss computation and optimization."""

from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_collections.config_dict import ConfigDict

from .models import utils as mutils
from .sde_lib import SDE, VESDE, VPSDE


def get_optimizer(config: ConfigDict, params) -> optim.Optimizer:
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported yet!")

    return optimizer


def optimization_manager(config: ConfigDict):
    """
    Creates and returns an optimization function based on the given configuration.

    The returned function applies learning rate warmup, gradient clipping,
    and performs an optimizer step.

    Args:
        config (ConfigDict): A configuration object containing the following attributes:
            - optim.lr (float): Initial learning rate.
            - optim.warmup (int): Number of steps for linear learning rate warmup.
            - optim.grad_clip (float): Maximum gradient norm for clipping (disabled if negative).

    Returns:
        Callable: A function to perform optimization with warmup and gradient clipping.
    """

    def optimize_fn(
        optimizer: optim.Optimizer,
        params: list,
        step: int,
        lr: float = config.optim.lr,
        warmup: int = config.optim.warmup,
        grad_clip: float = config.optim.grad_clip,
    ) -> None:
        """
        Performs a single optimization step with optional learning rate warmup
        and gradient clipping.

        During the warmup phase, the learning rate is scaled linearly from 0
        to the specified learning rate (`lr`) over the `warmup` steps.
        Gradient clipping constrains the maximum norm of gradients if enabled.

        Args:
            optimizer: The optimizer instance to update parameters.
            params: The list of parameters to optimize.
            step: The current training step, used for warmup scheduling.
            lr: Base learning rate. Defaults to config.optim.lr.
            warmup: Number of steps for linear warmup. Defaults to config.optim.warmup.
            grad_clip: Maximum norm for gradient clipping. Disabled if negative.
                Defaults to config.optim.grad_clip.

        Returns:
            None
        """
        # Learning rate warmup
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)

        # Gradient clipping
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        optimizer.step()

    return optimize_fn


def get_deterministic_loss_fn(train, reduce_mean=True):
    def loss_fn(model, batch, cond, generator=None):
        """Compute the loss function for a deterministic run.

        Args:
          model: A score model.
          batch: A mini-batch of training/evaluation data to model.
          cond: A mini-batch of conditioning inputs.
          generator: An optional random number generator so can control the timesteps and initial noise samples used by loss function [ignored in train mode]

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        # for deterministic model, do not use the time or target inputs - set to 0 always
        x = torch.zeros_like(batch)
        t = torch.zeros(batch.shape[0], device=batch.device)
        pred = model(x, cond, t)
        loss = F.mse_loss(pred, batch, reduction="mean")
        return loss

    return loss_fn


def get_sde_loss_fn(
    sde: Type[SDE],
    train: bool,
    reduce_mean: bool = True,
    continuous: bool = True,
    likelihood_weighting: bool = True,
    eps: float = 1e-5,
):
    """Creates a loss function for training or evaluating with arbitrary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss
            across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
            Otherwise it requires ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        loss_fn: A callable `loss_fn` that computes the loss for a given model, batch, and
        conditioning inputs.
    """
    reduce_op = (
        torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(
        model: torch.nn.Module, batch: torch.Tensor, cond: torch.Tensor, generator=None
    ) -> torch.Tensor:
        """Estimates the score and computes the loss.

        Args:
            model: A score model.
            batch: A mini-batch of training/evaluation data to model.
            cond: A mini-batch of conditioning inputs.
            generator: An optional random number generator for deterministic sampling during
                evaluation.

        Returns:
            loss: The average loss value across the mini-batch.
        """
        # Retrieve the score function
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

        # Sample time steps and noise
        if train:
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            z = torch.randn_like(batch)
        else:
            t = (
                torch.rand(batch.shape[0], device=batch.device, generator=generator) * (sde.T - eps)
                + eps
            )
            z = torch.empty_like(batch).normal_(generator=generator)

        # Pertrub the input data using the SDE's marginal probability function (applies noise based
        # on the time step t).
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z

        # Compute estimate of the score.
        score = score_fn(perturbed_data, cond, t)

        # Compute the loss.
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        # Average the loss across the batch.
        loss = torch.mean(losses)

        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = (
        torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas**2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = (
            sqrt_alphas_cumprod[labels, None, None, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        )
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(
    sde: Type[SDE],
    train: bool,
    optimize_fn: callable,
    reduce_mean: bool = False,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    deterministic: bool = False,
):
    """Returns a function to perform a single training or evaluation step.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: Whether the step function is for training (True) or evaluation (False).
        optimize_fn: A callable that performs a single optimization step. It should take the
            following arguments:
            - optimizer (torch.optim.Optimizer): The optimizer instance to update parameters.
            - params (list): The list of parameters to optimize.
            - step (int): The current training step.
            - lr (float, optional): Learning rate.
            - warmup (int, optional): Number of steps for linear learning rate warmup.
            - grad_clip (float, optional): Maximum gradient norm for clipping.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss
            across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
        deterministic: If true, use deterministic mode loss, else use diffusion losses.

    Returns:
         A function that performs a single training or evaluation step, depending on the
         `train` flag.
    """
    # Get the loss function. The loss function returns the average loss over a mini-batch.
    if deterministic:
        loss_fn = get_deterministic_loss_fn(train, reduce_mean=reduce_mean)
    else:
        if continuous:
            loss_fn = get_sde_loss_fn(
                sde,
                train,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, VESDE):
                loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
            elif isinstance(sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

    def step_fn(
        state: dict, batch: torch.Tensor, cond: torch.Tensor, generator=None
    ) -> torch.Tensor:
        """Runs a single training or evaluation step.

        For training, this function will estimate the score, compute the loss, backpropagate
        the gradients, and perform an optimization step. For evaluation, it will only estimate
        the score a compute the loss (i.e., a forward pass).

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and
        jit-compiled together for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
                EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data to model.
            cond: A mini-batch of conditioning inputs.
            generator: An optional random number generator so can control the timesteps and initial
                noise samples used by loss function. Ignored in train mode.

        Returns:
            loss: The average loss value across the mini-batch (averaged over all samples in the
                mini-batch).
        """
        model = state["model"]

        if train:
            # Load the optimizer and zero the gradients
            optimizer = state["optimizer"]
            optimizer.zero_grad()

            # Forward pass (estimate scores and compute loss)
            loss = loss_fn(model, batch, cond)

            # Backpropagate the loss and perform an optimization step
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state["step"])
            state["step"] += 1
            state["ema"].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state["ema"]
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Forward pass (estimate scores and compute loss)
                loss = loss_fn(model, batch, cond, generator=generator)

                ema.restore(model.parameters())

        return loss

    return step_fn
