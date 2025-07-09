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
# Significant modifications to the original work have been made by Henry Addison
# to allow for conditional modelling, location-specific parameters,
# removal of tensorflow dependency, tracking for training via Weights and Biases
# and MLFlow, and iterating by epoch using PyTorch DataLoaders
"""Training for score-based generative models."""

import itertools
import logging
import os
from collections import defaultdict

import torch
import torchvision
import mlflow
from codetiming import Timer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ml_collections.config_dict import ConfigDict

from . import likelihood, losses, sampling, sde_lib

# Keep the import below for registering all model definitions
from .models import cncsnpp, cunet
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from .models.location_params import LocationParams
from .mlde_utils import restore_checkpoint, save_checkpoint
from mlde.data import get_dataloader
from mlde.mlde_utils import DatasetMetadata


def val_loss(
    config: ConfigDict, eval_dl: torch.utils.data.DataLoader, eval_step_fn: callable, state: dict
) -> float:
    """
    Compute the average loss on the validation set.

    Args:
        config: Configuration object.
        eval_dl: Dataloader for the validation set.
        eval_step_fn: A function that performs a single evaluation step.
        state: Dictionary containing the current state of the training loop.

    Returns:
        val_set_loss: Average loss over all samples in the validation set.
    """
    val_set_loss = 0.0

    # Use a consistent generator for computing validation set loss so value is not down to
    # vagaries of random choice of initial noise samples or schedules.
    g = torch.Generator(device=config.device)
    g.manual_seed(42)

    for eval_cond_batch, eval_target_batch, _ in eval_dl:
        eval_target_batch = eval_target_batch.to(config.device)
        eval_cond_batch = eval_cond_batch.to(config.device)

        # Append any location-specific parameters
        eval_cond_batch = state["location_params"](eval_cond_batch)

        # Runs a single evaluation step - estimates the score and computes the loss over a
        # mini-batch of data. Returns the average loss over all samples in the batch.
        eval_loss = eval_step_fn(state, eval_target_batch, eval_cond_batch, generator=g)

        val_set_loss += eval_loss.item()

    # Divide by the number of mini-batches to get the average loss over the validation set (per
    # sample).
    val_set_loss = val_set_loss / len(eval_dl)

    return val_set_loss


@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
def train(config: ConfigDict, datadir: str, workdir: str):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        datadir: Path to directory containing the datasets. Each dataset is expected to be a folder
            with associated train.nc, val.nc, test.nc and ds-config.yaml files.
        workdir: Path to working directory. Transforms and model checkpoints will be written here.
            If this contain checkpoiints, then training will be resumed from the latest checkpoint.
    """
    # Print contents of workdir
    # TODO: Rethink how workdir is used. Definitely need a way to start training from stratch.
    # Also, don't want different training runs interfering with each other.
    logging.info(f"Contents of workdir: {os.listdir(workdir)}")

    # Save the config
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        f.write(config.to_yaml())

    # Create transform saving directory
    transform_dir = os.path.join(workdir, "transforms")
    os.makedirs(transform_dir, exist_ok=True)

    # Create a dictionary of target transform keys. This is a mapping from target name (string) to the
    # name of the mapping (string) to be applied. Uses the default target transform key for all targets unless
    # an override is specified.
    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(
        config.data.target_transform_overrides
    )

    logging.info(f"Target transform keys: {dict(target_xfm_keys)}")
    mlflow.log_param("target_xfm_keys", dict(target_xfm_keys))

    logging.info(f"Config: \n{config}")

    run_config = dict(
        dataset=config.data.dataset_name,
        input_transform_key=config.data.input_transform_key,
        architecture=config.model.name,
        sde=config.training.sde,
        num_train_epochs=config.training.n_epochs,
    )
    mlflow.log_params(run_config)

    # Build dataloaders
    dataset_meta = DatasetMetadata(datadir, config.data.dataset_name)
    train_dl, _, _ = get_dataloader(
        datadir,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.input_transform_key,
        target_xfm_keys,
        transform_dir,
        batch_size=config.training.batch_size,
        split="train",
        ensemble_members=dataset_meta.ensemble_members(),
        include_time_inputs=config.data.time_inputs,
        evaluation=False,
    )
    eval_dl, _, _ = get_dataloader(
        datadir,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.input_transform_key,
        target_xfm_keys,
        transform_dir,
        batch_size=config.training.batch_size,
        split="val",
        ensemble_members=dataset_meta.ensemble_members(),
        include_time_inputs=config.data.time_inputs,
        evaluation=False,
        shuffle=False,
    )

    # Initialize model.
    score_model = mutils.create_model(config=config, datadir=datadir)

    # Include a learnable feature map
    location_params = LocationParams(config.model.loc_spec_channels, config.data.image_size)
    location_params = location_params.to(config.device)
    # location_params = torch.nn.DataParallel(location_params)

    ema = ExponentialMovingAverage(
        itertools.chain(score_model.parameters(), location_params.parameters()),
        decay=config.model.ema_rate,
    )

    optimizer = losses.get_optimizer(
        config, itertools.chain(score_model.parameters(), location_params.parameters())
    )

    state = dict(
        optimizer=optimizer,
        model=score_model,
        location_params=location_params,
        ema=ema,
        step=0,
        epoch=0,
    )

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    # TODO: Do we need this?
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    state, _ = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_epoch = int(state["epoch"]) + 1  # start from the epoch after the one currently reached

    # Setup SDEs
    deterministic = "deterministic" in config and config.deterministic
    if deterministic:
        sde = None
    else:
        if config.training.sde.lower() == "vpsde":
            sde = sde_lib.VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        elif config.training.sde.lower() == "subvpsde":
            sde = sde_lib.subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales,
            )
        elif config.training.sde.lower() == "vesde":
            sde = sde_lib.VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=config.model.num_scales,
            )
        else:
            raise NotImplementedError(
                f"SDE '{config.training.sde}' specified in the config (under training.sde) is "
                f"not implemented. Please choose from 'VPSDE', 'subVPSDE', or 'VESDE'."
            )

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        likelihood_weighting=config.training.likelihood_weighting,
        deterministic=deterministic,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        likelihood_weighting=config.training.likelihood_weighting,
        deterministic=deterministic,
    )

    num_train_epochs = config.training.n_epochs

    logging.info(f"Starting training loop at epoch {initial_epoch}.")

    if config.training.random_crop_size > 0:
        random_crop = torchvision.transforms.RandomCrop(config.training.random_crop_size)

    for epoch in range(initial_epoch, num_train_epochs + 1):
        state["epoch"] = epoch
        train_set_loss = 0.0
        with logging_redirect_tqdm():
            with tqdm(
                total=len(train_dl.dataset), desc=f"Epoch {state['epoch']}", unit="timesteps"
            ) as pbar:
                for cond_batch, target_batch, _ in train_dl:
                    target_batch = target_batch.to(config.device)
                    cond_batch = cond_batch.to(config.device)

                    # Append any location-specific parameters
                    cond_batch = state["location_params"](cond_batch)

                    if config.training.random_crop_size > 0:
                        x_ch = target_batch.shape[1]
                        cropped = random_crop(torch.cat([target_batch, cond_batch], dim=1))
                        target_batch = cropped[:, :x_ch]
                        cond_batch = cropped[:, x_ch:]

                    # Execute one training step (forward pass, backpropagate loss, and update
                    # model)
                    loss = train_step_fn(state, target_batch, cond_batch)
                    train_set_loss += loss.item()

                    # Log the training loss periodically
                    if state["step"] % config.training.log_freq == 0:
                        logging.info(
                            "epoch: %d, step: %d, train_loss: %.5e"
                            % (state["epoch"], state["step"], loss.item())
                        )
                        mlflow.log_metric(
                            "Training loss (average per sample, per step)",
                            loss.item(),
                            step=state["step"],
                        )

                    # Report the loss on an evaluation dataset periodically
                    if state["step"] % config.training.eval_freq == 0:
                        val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
                        logging.info(
                            "epoch: %d, step: %d, val_loss: %.5e"
                            % (state["epoch"], state["step"], val_set_loss)
                        )
                        mlflow.log_metric(
                            "Validation loss (average per sample, per step)",
                            val_set_loss,
                            step=state["step"],
                        )

                    # Log progress so far on epoch
                    pbar.update(cond_batch.shape[0])

        # Report the loss on the training and validation datasets each epoch
        train_set_loss = train_set_loss / len(
            train_dl
        )  # average loss per sample in the training set
        val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
        mlflow.log_metric(
            "Training loss (average per sample, per epoch)", train_set_loss, step=state["epoch"]
        )
        mlflow.log_metric(
            "Validation loss (average per sample, per epoch)", val_set_loss, step=state["epoch"]
        )

        # Save a temporary checkpoint to resume training after each epoch
        save_checkpoint(checkpoint_meta_dir, state)

        if (state["epoch"] != 0 and state["epoch"] % config.training.snapshot_freq == 0) or state[
            "epoch"
        ] == num_train_epochs:
            # Save the checkpoint.
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{state['epoch']}.pth")
            save_checkpoint(checkpoint_path, state)
            logging.info(f"epoch: {state['epoch']}, checkpoint saved to {checkpoint_path}")
