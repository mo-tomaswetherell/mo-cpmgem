import glob
import os
from pathlib import Path
from typing import List
import yaml

import cartopy.crs as ccrs
import cftime

# TODO: Globals should be capitalised.
cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()

DEFAULT_ENSEMBLE_MEMBER = "01"

TIME_PERIODS = {
    "historic": (
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2000, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
    "present": (
        cftime.Datetime360Day(2020, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2040, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
    "future": (
        cftime.Datetime360Day(2060, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
}
"""Time periods for the historic, present and future data."""


# TODO: This class is almost entirely unused .. only ensemble_members() is called.
class DatasetMetadata:
    """Metadata associated with a dataset."""

    def __init__(self, datadir: str, name: str):
        """Initialise DatasetMetadata.

        Args:
            datadir: Path to directory containing the dataset
            name: Name of the dataset, e.g. "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr".
                This is expected to be a folder within `datadir` containing the dataset files.
        """
        self.datadir = datadir
        self.name = name

    def __str__(self) -> str:
        return f"DatasetMetadata({self.path()})"

    def path(self) -> Path:
        """Path to the dataset directory."""
        return Path(self.datadir, self.name)

    def splits(self) -> List[str]:
        return map(
            lambda f: os.path.splitext(f)[0],
            glob.glob("*.nc", root_dir=str(self.path())),
        )

    def split_path(self, split: str) -> Path:
        """Path to a specific split of the dataset.

        Args:
            split: Name of the split, e.g. "train", "val", "test".
        """
        return self.path() / f"{split}.nc"

    def config_path(self) -> Path:
        """Path to the dataset configuration file.

        This file is expected to be named `ds-config.yml`, stored within the dataset folder,
        and contain metadata about the dataset.
        """
        return self.path() / "ds-config.yml"

    def config(self) -> dict:
        """Load the dataset configuration file."""
        with open(self.config_path(), "r") as f:
            return yaml.safe_load(f)

    def ensemble_members(self) -> List[str]:
        """List of ensemble members associated with the dataset."""
        return self.config()["ensemble_members"]


# TODO: Might not be needed.
def samples_path(
    workdir: str,
    checkpoint: str,
    input_xfm: str,
    dataset: str,
    split: str,
    ensemble_member: str,
) -> Path:
    """Path to directory containing generated samples."""
    return Path(workdir, "samples", checkpoint, dataset, input_xfm, split, ensemble_member)


def samples_glob(samples_path: Path) -> list[Path]:
    return samples_path.glob("predictions-*.nc")


def dataset_path(datadir: str, dataset: str) -> Path:
    """Path to the directory containing the dataset files.

    Args:
        datadir: Path to the directory containing the dataset folders.
        dataset: Name of the dataset, e.g. "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr".
            This is expected to be a folder within `datadir` containing the dataset files.
    """
    return Path(datadir, dataset)


def dataset_split_path(datadir: str, dataset: str, split: str) -> Path:
    """Path to a specific split of the dataset."""
    return dataset_path(datadir, dataset) / f"{split}.nc"


def dataset_config_path(datadir: str, dataset: str) -> Path:
    """Path to the dataset configuration file."""
    return dataset_path(datadir, dataset) / "ds-config.yml"


def dataset_config(datadir: str, dataset: str) -> dict:
    """Load the dataset configuration file."""
    with open(dataset_config_path(datadir, dataset), "r") as f:
        return yaml.safe_load(f)
