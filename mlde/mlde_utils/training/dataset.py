from datetime import timedelta
import gc
import logging
import os
from collections.abc import Callable

import xarray as xr

from ..transforms import (
    build_input_transform,
    build_target_transform,
    save_transform,
    load_transform,
)

from .. import dataset_split_path, dataset_config
from mlde.mlde_utils.transforms import ComposeT


def get_dataset(
    datadir: str,
    active_dataset_name: str,
    model_src_dataset_name: str,
    input_transform_dataset_name: str,
    input_transform_key: str,
    target_transform_keys: dict[str, str],
    transform_dir: str | None,
    split: str,
    ensemble_members: list[str],
    evaluation: bool = False,
) -> tuple[xr.Dataset, ComposeT, ComposeT]:
    """Return xarray.Dataset and transforms for a named dataset split.

    Args:
        datadir: Path to directory containing dataset folders.
        active_dataset_name: Name of dataset from which to load data splits
        model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
        input_transform_dataset_name: Name of dataset to use for fitting input transform (may be
            the same as active_dataset_name or model_src_dataset_name)
        input_transform_key: Name of input transform pipeline to use
        target_transform_keys: Mapping from name of target variable to name of target transform
            pipeline to use
        transform_dir: Path to directory where transforms are stored. If None, transforms will be
            created in memory and not written to disk.
        split: Split of the active dataset to load
        ensemble_members: Ensemble members to load
        evaluation: If `True`, don't allow fitting of target transform

    Returns:
      dataset, transform, target_transform
    """
    transform, target_transform = _find_or_create_transforms(
        datadir,
        input_transform_dataset_name,
        model_src_dataset_name,
        transform_dir,
        input_transform_key,
        target_transform_keys,
        evaluation,
    )

    xr_data = load_raw_dataset_split(datadir, active_dataset_name, split).sel(
        ensemble_member=ensemble_members
    )

    xr_data = transform.transform(xr_data)
    xr_data = target_transform.transform(xr_data)

    return xr_data, transform, target_transform


def open_raw_dataset_split(datadir: str, dataset_name: str, split: str) -> xr.Dataset:
    """
    Returns an xarray dataset for a given dataset split.

    Calls `xr.open_dataset` on the dataset split file.

    Args:
        dataset_name: Name of the dataset
        split: Name of the split, e.g. "train", "val", "test"
    """
    return xr.open_dataset(dataset_split_path(datadir, dataset_name, split))


def load_raw_dataset_split(datadir: str, dataset_name: str, split: str) -> xr.Dataset:
    """
    Returns an xarray dataset for a given dataset split.

    Calls `xr.load_dataset` on the dataset split file. This differs from `xr.open_dataset` in that
    it "loads the Dataset into memory, closes the file, and returns the Dataset. In contrast,
    open_dataset keeps the file handle open and lazy loads its content.

    Args:
        datadir: Path to directory containing dataset folders.
        dataset_name: Name of the dataset
        split: Name of the split, e.g. "train", "val", "test"
    """
    return xr.load_dataset(dataset_split_path(datadir, dataset_name, split))


def get_variables(datadir: str, dataset_name: str) -> tuple[list[str], list[str]]:
    """
    Returns the names of the predictor variables and target variables from a dataset.

    Args:
        datadir: Path to directory containing dataset folders.
        dataset_name: Name of the dataset. There is expected to be a folder in `datadir` with the
            same name as `dataset_name`.

    Returns:
        variables: List of predictor variables, e.g. ["psl", "spechum250", "temp850"].
        target_variables: List of target variable. Each will be prefixed with "target_".
    """
    ds_config = dataset_config(datadir=datadir, dataset=dataset_name)

    variables = ds_config["predictors"]["variables"]
    target_variables = list(map(lambda v: f"target_{v}", ds_config["predictands"]["variables"]))

    return variables, target_variables


def _build_transform(
    datadir: str,
    variables: list[str],
    active_dataset_name: str,
    model_src_dataset_name: str,
    transform_keys: str | dict[str, str],
    builder: Callable[[list[str], str | dict[str, str]], ComposeT],
) -> ComposeT:
    """Create and fit a transform pipeline.

    Args:
        datadir: Path to directory containing datasets.
        variables: List of variables.
        active_dataset_name: Name of dataset from which to load data splits.
        model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same).
        transform_keys:
        builder:

    Returns:
        xfm: Fitted transform pipeline.
    """
    logging.info(f"Fitting transform")

    xfm = builder(variables, transform_keys)

    model_src_training_split = open_raw_dataset_split(datadir, model_src_dataset_name, "train")
    active_dataset_training_split = open_raw_dataset_split(datadir, active_dataset_name, "train")

    xfm.fit(active_dataset_training_split, model_src_training_split)

    model_src_training_split.close()
    del model_src_training_split
    active_dataset_training_split.close()
    del active_dataset_training_split
    gc.collect

    return xfm


def _find_or_create_transforms(
    datadir: str,
    active_dataset_name: str,
    model_src_dataset_name: str,
    transform_dir: str | None,
    input_transform_key: str,
    target_transform_keys: dict[str, str],
    evaluation: bool,
) -> tuple[ComposeT, ComposeT]:
    """Return input and target transforms.

    Args:
        datadir: Path to directory containing datasets.
        active_dataset_name: Name of dataset from which to load data splits
        model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
        transform_dir: Path to directory where transforms are stored. If None, transforms will be
            created in memory and not written to disk.
        input_transform_key: Name of input transform pipeline to use
        target_transform_keys: Mapping from name of target variable to name of target transform
            pipeline to use
        evaluation: If `True`, don't allow fitting of target transform

    Returns:
        input_transform: Fitted input transform
        target_transform: Fitted target transform

    Raises:
        RuntimeError: If target transform is attempted to be fitted during evaluation.
    """
    variables, target_variables = get_variables(datadir, model_src_dataset_name)

    if transform_dir is None:
        input_transform = _build_transform(
            datadir,
            variables,
            active_dataset_name,
            model_src_dataset_name,
            input_transform_key,
            build_input_transform,
        )

        if evaluation:
            raise RuntimeError("Target transform should only be fitted during training")
        target_transform = _build_transform(
            datadir,
            target_variables,
            active_dataset_name,
            model_src_dataset_name,
            target_transform_keys,
            build_target_transform,
        )
    else:
        dataset_transform_dir = os.path.join(
            transform_dir, active_dataset_name, input_transform_key
        )
        os.makedirs(dataset_transform_dir, exist_ok=True)
        input_transform_path = os.path.join(dataset_transform_dir, "input.pickle")
        target_transform_path = os.path.join(transform_dir, "target.pickle")

        if os.path.exists(input_transform_path):
            input_transform = load_transform(input_transform_path)
        else:
            input_transform = _build_transform(
                datadir,
                variables,
                active_dataset_name,
                model_src_dataset_name,
                input_transform_key,
                build_input_transform,
            )
            save_transform(input_transform, input_transform_path)

        if os.path.exists(target_transform_path):
            target_transform = load_transform(target_transform_path)
        else:
            if evaluation:
                raise RuntimeError("Target transform should only be fitted during training")
            target_transform = _build_transform(
                datadir,
                target_variables,
                active_dataset_name,
                model_src_dataset_name,
                target_transform_keys,
                build_target_transform,
            )
            save_transform(target_transform, target_transform_path)

    gc.collect
    return input_transform, target_transform
