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
python train.py \
  --config=<full path to config file> \
  --datadir=<full path to directory containing the dataset specified in the config> \
  --workdir=<path to directory to write outputs to>
```
"""

print("Starting train.py script.")

import logging

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

logger.info("Set up logger. About to import libraries.")

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

logger.info("Finished importing libraries. About to import run_lib.")

import mlde.run_lib as run_lib

logger.info("Finished importing run_lib.")


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
flags.mark_flags_as_required(["config", "datadir", "workdir"])


def main(_):
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"Config: \n{FLAGS.config.to_dict()}")
    logger.info(f"datadir: {FLAGS.datadir}")
    logger.info(f"workdir: {FLAGS.workdir}")

    # Start training
    logger.info("Starting training with run_lib.train()")
    run_lib.train(FLAGS.config, FLAGS.datadir, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
