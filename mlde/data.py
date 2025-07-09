"""Loading UKCP Local data into PyTorch"""

from typing import Type

import cftime
import numpy as np
import torch
import xarray as xr
from mlde.mlde_utils.training.dataset import get_dataset, get_variables
from torch.utils.data import DataLoader, Dataset

from mlde.mlde_utils.transforms import ComposeT

TIME_RANGE = (
    cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
    cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True),
)


class UKCPLocalDataset(Dataset):
    """Custom Dataset for UKCP Local data."""

    def __init__(
        self,
        ds: xr.Dataset,
        variables: list[str],
        target_variables: list[str],
        time_range: tuple[cftime.Datetime360Day, cftime.Datetime360Day] | None,
    ):
        """Initialise UKCPLocalDataset.

        Args:
            ds: xarray Dataset containing UKCP Local data
            variables: List of variable names to use as input (predictor) features
            target_variables: List of variable names to use as targets (predictands)
            time_range: Tuple of start and end times for the data. If not None, time features will be included in the input data.
        """
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range

    @classmethod
    def variables_to_tensor(cls, ds, variables) -> torch.Tensor:
        return torch.tensor(
            # stack features before lat-lon (HW)
            np.stack([ds[var].values for var in variables], axis=-3)
        ).float()

    @classmethod
    def time_to_tensor(cls, ds, shape, time_range) -> torch.Tensor:
        climate_time = np.array(ds["time"] - time_range[0]) / np.array(
            [time_range[1] - time_range[0]], dtype=np.dtype("timedelta64[ns]")
        )
        season_time = ds["time.dayofyear"].values / 360

        return (
            torch.stack(
                [
                    torch.tensor(climate_time).broadcast_to((climate_time.shape[0], *shape[-2:])),
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                ],
                dim=-3,
            )
            .squeeze()
            .float()
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.ds.time) * len(self.ds.ensemble_member)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Return the input and target variables for the given index.

        Args:
            idx: Index of the sample to return

        Returns:
            cond: Input features
            x: Target features
            time: Time.
        """
        subds = self.sel(idx)

        cond = self.variables_to_tensor(subds, self.variables)
        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)

        time = subds["time"].values.reshape(-1)

        return cond, x, time

    def sel(self, idx) -> xr.Dataset:
        """Return a subset of the dataset corresponding to the given index."""
        em_idx, time_idx = divmod(idx, len(self.ds.time))
        return self.ds.isel(time=time_idx, ensemble_member=em_idx)


def build_dataloader(
    xr_data: xr.Dataset,
    variables: list[str],
    target_variables: list[str],
    batch_size: int,
    shuffle: bool,
    include_time_inputs: bool,
) -> DataLoader:
    def custom_collate(batch):
        from torch.utils.data import default_collate

        return *default_collate([(e[0], e[1]) for e in batch]), np.concatenate(
            [e[2] for e in batch]
        )

    time_range = None
    if include_time_inputs:
        time_range = TIME_RANGE
    xr_dataset = UKCPLocalDataset(xr_data, variables, target_variables, time_range)
    data_loader = DataLoader(
        xr_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate
    )
    return data_loader


def get_dataloader(
    datadir: str,
    active_dataset_name: str,
    model_src_dataset_name: str,
    input_transform_dataset_name: str,
    input_transform_key: str,
    target_transform_keys: dict[str, str],
    transform_dir: str,
    batch_size: int,
    split: str,
    ensemble_members: list[str],
    include_time_inputs,
    evaluation: bool = False,
    shuffle=True,
) -> tuple[DataLoader, ComposeT, ComposeT]:
    """Create Dataloader for given dataset and split.

    Args:
        datadir: Path to directory containing datasets
        active_dataset_name: Name of dataset from which to load data splits
        model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
        input_transform_dataset_name: Name of dataset to use for fitting input transform (may be
            the same as active_dataset_name or model_src_dataset_name)
        transform_dir: Path to where transforms should be stored
        input_transform_key: Name of input transform pipeline to use
        target_transform_keys: Mapping from target variable name to target transform pipeline to use
        transform_dir: Path to where transforms should be stored
        batch_size: Size of batch to use for DataLoaders
        split: Split of the active dataset to load
        ensemble_members: Ensemble members to load. For example, ["01", "02", "03"]
        include_time_inputs:
        evaluation: If `True`, fix number of epochs to 1.
        shuffle: If `True`, shuffle the data after each epoch.

    Returns:
      data_loader, transform, target_transform.
    """
    xr_data, transform, target_transform = get_dataset(
        datadir,
        active_dataset_name,
        model_src_dataset_name,
        input_transform_dataset_name,
        input_transform_key,
        target_transform_keys,
        transform_dir,
        split,
        ensemble_members,
        evaluation,
    )

    variables, target_variables = get_variables(datadir, model_src_dataset_name)

    data_loader = build_dataloader(
        xr_data,
        variables,
        target_variables,
        batch_size,
        shuffle,
        include_time_inputs,
    )

    return data_loader, transform, target_transform


def np_samples_to_xr(
    np_samples: np.ndarray,
    target_transform: Type[ComposeT],
    target_vars: list[str],
    coords,
    cf_data_vars: dict,
) -> xr.Dataset:
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any
    transformation applied to the target variables before modelling.

    Args:
        np_samples:
        target_transform:
        target_vars: List of target variable names (e.g. ["target_precipitation_rate",
            "target_temperature"])
        coords:
        cf_data_vars:

    Returns:
        samples_ds: xarray Dataset containing the samples
    """
    coords = {**dict(coords)}
    data_vars = {**cf_data_vars}
    pred_dims = ["ensemble_member", "time", "grid_latitude", "grid_longitude"]

    for var_idx, var in enumerate(target_vars):
        # Add ensemble member axis to np samples and get just values for current variable
        np_var_pred = np_samples[np.newaxis, :, var_idx, :]

        pred_attrs = {
            "grid_mapping": "rotated_latitude_longitude",
            "standard_name": var.replace("target_", "pred_"),
        }
        pred_var = (pred_dims, np_var_pred, pred_attrs)
        raw_pred_var = (
            pred_dims,
            np_var_pred,
            {"grid_mapping": "rotated_latitude_longitude"},
        )
        data_vars.update(
            {
                var: pred_var,  # Don't rename pred var until after inverting target transform
                var.replace("target_", "raw_pred_"): raw_pred_var,
            }
        )

    samples_ds = target_transform.invert(xr.Dataset(data_vars=data_vars, coords=coords, attrs={}))
    samples_ds = samples_ds.rename({var: var.replace("target_", "pred_") for var in target_vars})

    for var_idx, var in enumerate(target_vars):
        pred_attrs = {
            "grid_mapping": "rotated_latitude_longitude",
            "standard_name": var.replace("target_", "pred_"),
            "units": "kg m-2 s-1",  # Assuming target is precipitation
        }
        samples_ds[var.replace("target_", "pred_")] = samples_ds[
            var.replace("target_", "pred_")
        ].assign_attrs(pred_attrs)

    return samples_ds
