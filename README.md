# mo-mlde

This is a fork of Henry Addison's [mlde](https://github.com/henryaddison/mlde) repository.

## Setup

Clone the repository:

```bash
git clone git@github.com:mo-tomaswetherell/mo-cpmgem.git
cd mo-cpmgem
```

Setup the environment (installs dependencies and `mlde` package):

```bash
conda env create -f environment.yaml
conda activate mo-cpmgem
```

## Training

Start an MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

This will be used to track/monitor the experiment.

In another terminal tab, run the training script:

```bash
python train.py \
  --config=<full path to config file> \
  --datadir=<full path to dataset directory> \
  --workdir=<path to directory to write outputs to>
```

Notes:
* To run training with the same configuration as used in the paper, use the config file `ukcp_local_pr_12em_cncsnpp_continuous.py` (the full path to this file should be given).
* `datadir` should be the full path to the directory *containing* the dataset folder(s). For example, if you want to use the dataset `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` (i.e., if this is the dataset specified in your chosen configuration), then `datadir` should be the full path to the folder which *contains* the directory `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` (and not the full path to `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` itself).