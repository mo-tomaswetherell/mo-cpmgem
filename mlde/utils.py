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
# to allow for location-specific parameters and iterating by epoch using PyTorch
# DataLoaders and helpers for determining a model size.

import logging
import os
import yaml
from pathlib import Path

import torch
from ml_collections.config_dict import ConfigDict


def load_config(config_path: str | Path) -> ConfigDict:
    """Load configuration from a yaml file.

    Args:
        config_path: Path to configuration yaml file.

    Returns:
        config: Configuration object. Loaded from the yaml file.
    """
    with open(config_path) as f:
        config = ConfigDict(yaml.unsafe_load(f))

    return config


def restore_checkpoint(
    ckpt_dir: str | Path, state: dict, device: torch.device
) -> tuple[dict, bool]:
    """Load the state of the model, optimizer, and EMA from a checkpoint file on disk.

    Args:
        ckpt_dir: Path to the checkpoint file, e.g. '/data/checkpoint.pth'.
        state: Dictionary containing the current state of the model, optimizer, EMA, step, epoch,
            and location_params. If the checkpoint file is found, the state will be updated with
            the loaded values. Otherwise, the state will be returned as is.
        device: Device to load the model to.

    Returns:
        state: Dictionary containing the loaded state of the model, optimizer, EMA, step, epoch,
            and location_params. If the checkpoint file is not found, the state will be returned
            as is.
        Boolean indicating whether the checkpoint file was found.
    """
    if not os.path.exists(ckpt_dir):
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        return state, False
    else:
        # Update the state with the loaded values
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["location_params"].load_state_dict(loaded_state["location_params"])
        state["step"] = loaded_state["step"]
        state["epoch"] = loaded_state["epoch"]
        logging.info(
            f"Checkpoint found at {ckpt_dir}. "
            f"Returned the state from {state['epoch']}/{state['step']}"
        )
        return state, True


def save_checkpoint(ckpt_dir: str | Path, state: dict):
    """Save the state of the model, optimizer, and EMA to a checkpoint file on disk.

    Args:
        ckpt_dir: Path to save the checkpoint file, e.g. '/data/checkpoint.pth'.
        state: Dictionary containing the model, optimizer, EMA, step, epoch, and location_params.
            Expected keys are 'model', 'optimizer', 'ema', 'step', 'epoch', and 'location_params'.
    """
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
        "location_params": state["location_params"].state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def param_count(model: torch.nn.Module) -> int:
    """Return the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters())


def model_size(model: torch.nn.Module) -> float:
    """Compute size in memory of model in MB."""
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())

    return (param_size + buffer_size) / 1024**2
