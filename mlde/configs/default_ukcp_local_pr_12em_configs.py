"""Default config for training model on UKCP Local precip data with 1 ensemble member."""

from mlde.configs.default_ukcp_local_pr_1em_configs import (
    get_default_configs as get_base_configs,
)


def get_default_configs():
    """Get the default configuration."""
    config = get_base_configs()

    # Training
    training = config.training
    training.n_epochs = 20
    training.snapshot_freq = 5
    training.eval_freq = 5000

    # Data
    data = config.data
    data.dataset_name = "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr"

    return config
