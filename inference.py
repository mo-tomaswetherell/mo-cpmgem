"""Generate samples using a trained diffusion model."""

import os
import argparse
import logging
from collections import defaultdict
from pathlib import Path

from ml_collections.config_dict import ConfigDict

from mlde.data import get_dataloader
from mlde.inference import load_model, sample
from mlde.utils import load_config


# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        args: Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained score-based model."
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help=(
            "Path to configuration yaml file that was used for training the model "
            "specified by `model_checkpoint_path`."
        ),
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--transformdir",
        type=str,
        required=True,
        help="Path to directory containing transform files.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Path to working directory. Samples are saved to `workdir/samples`.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Path to directory containing datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Name of the dataset to load the conditioning input data from, i.e. the dataset to "
            "sample from."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help=(
            "Split to load from the dataset. Expected to be one of 'train', 'val', 'test'."
            "Samples are generated for this split."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Batch size to use for sampling. If not provided, the value from the configuration "
            "file will be used. A larger batch size will use more memory and will speed up sampling."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate for each timestep.",
    )
    parser.add_argument(
        "--input_transform_dataset",
        type=str,
        default=None,
        help=(
            "Name of the dataset that was used to fit the input transform (may be the same as"
            " 'dataset'). If not provided, `dataset` will be used (assumes you are sampling "
            "from the same dataset that was used to fit the input transform, i.e. the dataset "
            "used in training - though not necessarily the same split)."
        ),
    )
    parser.add_argument(
        "--input_transform_key",
        type=str,
        default=None,
        help=(
            "Name of the input transform pipeline to use. If not provided, the value from the "
            "configuration file will be used."
        ),
    )
    parser.add_argument(
        "--ensemble_member",
        type=str,
        default="01",
        help="Ensemble member to load the conditioning input data from.",
    )

    args = parser.parse_args()

    return args


def main(
    config_path: str | Path,
    model_checkpoint_path: str | Path,
    transformdir: str | Path,
    workdir: str | Path,
    datadir: str | Path,
    dataset: str,
    split: str = "val",
    batch_size: int | None = None,
    num_samples: int = 3,
    input_transform_dataset: str | None = None,
    input_transform_key: str | None = None,
    ensemble_member: str = "01",
):
    """Generate samples using a trained diffusion model.

    The generated samples are saved to disk as netCDF files in `workdir/samples`.

    Args:
        config_path: Path to configuration yaml file that was used for training the model
            specified by `model_checkpoint_path`.
        model_checkpoint_path: Path to model checkpoint file.
        transformdir: Path to directory containing transform files. The transform file(s) are
            expected to be .pickle files, created during the training process.
        workdir: Path to working directory. Samples are saved to `workdir/samples`.
        datadir: Path to directory containing datasets.
        dataset: Name of the dataset to load the conditioning input data from, i.e. the dataset
            to sample from.
        split: Split to load from the dataset. Expected to be one of "train", "val", "test".
            Samples are generated for this split.
        batch_size: Batch size to use for sampling. If not provided, the value from the
            configuration file will be used. A larger batch size will use more memory and will
            speed up sampling.
        num_samples: Number of samples to generate for each timestep.
        input_transform_dataset: Name of the dataset that was used to fit the input transform (may
            be the same as `dataset`). If not provided, `dataset` will be used (assumes you are
            sampling from the same dataset that was used to fit the input transform, i.e. the
            dataset used in training - though not necessarily the same split).
        input_transform_key: Name of the input transform pipeline to use. If not provided, the
            value from the configuration file will be used.
        ensemble_member: Ensemble member to load the conditioning input data from.
    """
    # Load config and update if overridden by function arguments
    config = load_config(config_path)

    if batch_size is not None:
        config.eval.batch_size = batch_size

    with config.unlocked():
        if input_transform_dataset is not None:
            config.data.input_transform_dataset = input_transform_dataset
        else:
            # Assumes you are sampling from the same dataset that was used to fit the input
            # transform (i.e. the dataset used in training - though not necessarily the same split).
            config.data.input_transform_dataset = dataset

        if "target_transform_overrides" not in config.data:
            config.data.target_transform_overrides = ConfigDict()

    if input_transform_key is not None:
        config.data.input_transform_key = input_transform_key

    # Save configuration to file.
    with open(os.path.join(workdir, "config.yml"), "w") as f:
        f.write(config.to_yaml())

    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(
        config.data.target_transform_overrides
    )

    # Create dataloader.
    eval_dl, _, target_transform = get_dataloader(
        datadir,
        dataset,
        config.data.dataset_name,
        config.data.input_transform_dataset,
        config.data.input_transform_key,
        target_xfm_keys,
        transformdir,
        split=split,
        ensemble_members=[ensemble_member],
        include_time_inputs=config.data.time_inputs,
        evaluation=True,
        batch_size=config.eval.batch_size,
        shuffle=False,
    )

    logger.info(f"Loading model from {model_checkpoint_path}")
    state, sampling_fn, target_vars = load_model(config, model_checkpoint_path, datadir)

    # Create `num_samples` set of samples for each timestep.
    for sample_id in range(num_samples):
        logger.info(f"Starting sample run {sample_id+1}/{num_samples}")
        xr_samples = sample(sampling_fn, state, config, eval_dl, target_transform, target_vars)

        # Save samples to disk.
        samples_path = f"{workdir}/samples"
        os.makedirs(samples_path, exist_ok=True)
        output_filepath = f"{samples_path}/predictions-{sample_id}.nc"
        logger.info(f"Saving samples to {output_filepath}")
        for varname in target_vars:
            for prefix in ["pred_", "raw_pred_"]:
                xr_samples[varname.replace("target_", prefix)].encoding.update(
                    zlib=True, complevel=5
                )
        xr_samples.to_netcdf(output_filepath)


if __name__ == "__main__":
    """Generate samples using a trained score-based model."""
    args = parse_args()
    logger.info(f"Arguments: {args}")
    main(
        args.config_path,
        args.model_checkpoint_path,
        args.transformdir,
        args.workdir,
        args.datadir,
        args.dataset,
        args.split,
        args.batch_size,
        args.num_samples,
        args.input_transform_dataset,
        args.input_transform_key,
        args.ensemble_member,
    )
