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

Set a working directory. All outputs (model checkpoints, mlflow logs, etc.) will be written here. It must be an absolute path .

```bash
export WORKDIR="full/path/to/workdir"
```

Run the training script:

```bash
python train.py \
  --config=<full path to config file> \
  --datadir=<full path to dataset directory> \
  --workdir="$WORKDIR"
```

If you're running on a remote server (e.g., JASMINE), from a terminal on the server start the MLflow UI:

```bash
mlflow ui --backend-store-uri file:$WORKDIR/mlruns --port 5000
```

You'll need the `mo-cpmgem` `conda` environment activated to use the `mlflow ui` command. Leave this process running.

Then, on your local machine, tunnel the MLflow UI from the remove server:

```bash
ssh -L 5000:localhost:5000 <your-username@remote-server-address>
```

Open your browser to `http://localhost:5000` to see live updates: params, metrics, etc.

**Notes:**
* To run training with the same configuration as used in the paper, use the config file `ukcp_local_pr_12em_cncsnpp_continuous.py` (the full path to this file should be given).
* `datadir` should be the full path to the directory *containing* the dataset folder(s). For example, if you want to use the dataset `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` (i.e., if this is the dataset specified in your chosen configuration), then `datadir` should be the full path to the folder which *contains* the directory `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` (and not the full path to `bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` itself).