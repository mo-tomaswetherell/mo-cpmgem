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
"""
Training script for CPMGEM.

Example use:

```bash
mlflow server --host 127.0.0.1 --port 5000

python train.py \
  --config=<full path to config file> \
  --datadir=<full path to directory containing the dataset specified in the config> \
  --workdir=<path to directory to write outputs to>
```
"""

import logging

import mlflow
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import mlde.run_lib as run_lib


# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config",
    None,
    (
        "Path to training config file. To train the model used in the paper use the config file "
        "ukcp_local_pr_12em_cncsnpp_continuous.py"
    ),
    lock_config=True,
)
flags.DEFINE_string(
    "datadir",
    None,
    (
        "Path to directory containing dataset folders. Each dataset folder is expected to "
        "contain the files train.nc, val.nc, test.nc and ds-config.yaml. Only the train.nc and "
        "the config file are required for training.",
    ),
)
flags.DEFINE_string(
    "workdir",
    None,
    "Path to working directory. All outputs (checkpoints, transforms) are stored here.",
)
flags.DEFINE_string(
    "mlflow_port",
    "5000",
    ("Port on which the MLflow server is running. Defaults to 5000."),
)
flags.DEFINE_string(
    "mlflow_experiment",
    "cpmgem",
    "Name of the experiment. Used to group runs in MLflow.",
)
flags.DEFINE_string(
    "mlflow_run_name",
    None,
    (
        "Name of the run. If not specified, a default name will be generated based on the "
        "current timestamp."
    ),
)
flags.mark_flags_as_required(["config", "datadir", "workdir"])


def main(_):
    # Setup MLflow tracking
    mlflow.set_experiment(FLAGS.mlflow_experiment)
    mlflow.set_tracking_uri(f"http://127.0.0.1:{FLAGS.mlflow_port}")
    with mlflow.start_run(run_name=FLAGS.mlflow_run_name):
        # Log the configuration
        mlflow.log_params(FLAGS.config.to_dict())
        mlflow.log_param("datadir", FLAGS.datadir)
        mlflow.log_param("workdir", FLAGS.workdir)

        # Start training
        run_lib.train(FLAGS.config, FLAGS.datadir, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
