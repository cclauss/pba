"""Train and evaluate models using augmentation schedules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import ray
from ray.tune import Trainable
from ray.tune import grid_search, run_experiments

from train_cifar import CifarModelTrainer
from train import RayModel


def main(_):
    from setup import create_parser, create_hparams
    FLAGS = create_parser("train")
    hparams = create_hparams("train", FLAGS)

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams.values(),
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": 5
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    train_spec["config"]["lr"] = 0.2
    train_spec["config"]["wd"] = grid_search([0.001, 0.002])

    # train_spec["config"]["lr"] = grid_search([0.25, 0.5, 1.0, 2.0])
    # train_spec["config"]["wd"] = grid_search([0.0005, 0.001, 0.00125, 0.0025, 0.005, 0.01])

    # train_spec["config"]["lr"] = grid_search([0.0125, 0.025, 0.05, 0.1, 0.2])
    # train_spec["config"]["wd"] = grid_search([0.0005, 0.00125, 0.0025, 0.005, 0.01, 0.02])

    ray.init()
    run_experiments({FLAGS.name: train_spec}, verbose=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
