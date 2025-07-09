"""Default config for training model on UKCP Local precip data with 1 ensemble member."""

import ml_collections
import torch


def get_default_configs():
    """Get the default configuration."""
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 16
    training.n_epochs = 100
    training.snapshot_freq = 25
    training.log_freq = 50
    training.eval_freq = 1000
    training.snapshot_freq_for_preemption = 1000  # store additional checkpoints for preemption in cloud computing environments. TODO: Look into what this actually means.
    training.snapshot_sampling = False  # produce samples at each snapshot.
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.random_crop_size = 0

    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # Evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 128
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "UKCP_Local"
    data.dataset_name = "bham64_ccpm-4x_1em_psl-sphum4th-temp4th-vort4th_pr"  # name of the dataset used to train the model
    data.image_size = 64
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.input_transform_dataset = None
    data.input_transform_key = "stan"
    data.target_transform_key = "sqrturrecen"
    data.target_transform_overrides = ml_collections.ConfigDict()
    data.time_inputs = False

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.1
    model.embedding_type = "fourier"
    model.loc_spec_channels = 0

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config.deterministic = False

    return config
