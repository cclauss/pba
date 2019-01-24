"""Run PBA Search."""

import contextlib
import os
import time
import random

import custom_ops as ops
import data_utils
import helper_utils
import numpy as np
import tensorflow as tf

import ray
from ray.tune import Trainable
from ray.tune import grid_search, run_experiments
from ray.tune.schedulers import PopulationBasedTraining

from train import RayModel
from train_cifar import CifarModelTrainer


def main(_):
    from setup import create_parser, create_hparams
    FLAGS = create_parser("search")
    hparams = create_hparams("search", FLAGS)
    hparams_config = hparams.values()

    train_spec = {
        "run": RayModel,
        "trial_resources": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams_config,
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": FLAGS.num_samples
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    # Custom explore func
    def explore(config):
        new_params = []
        if config["explore"] == "cifar10":
            for i, param in enumerate(config["hp_policy"]):
                if random.random() < 0.2:
                    if i % 2 == 0:
                        new_params.append(random.randint(0, 10))
                    else:
                        new_params.append(random.randint(0, 9))
                else:
                    amt = np.random.choice(
                        [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                    if random.random() < 0.5:
                        new_params.append(max(0, param - amt))
                    else:
                        if i % 2 == 0:
                            new_params.append(min(10, param + amt))
                        else:
                            new_params.append(min(9, param + amt))
        else:
            raise ValueError()
        config["hp_policy"] = new_params
        return config

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="val_acc",
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore)

    run_experiments({FLAGS.name: train_spec}, scheduler=pbt, verbose=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
