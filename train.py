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


class RayModel(Trainable):
    def _setup(self, *args):
        tf.logging.info('calling setup')
        self.hparams = tf.contrib.training.HParams(**self.config)
        self.trainer = CifarModelTrainer(self.hparams)

    def _train(self):
        """Runs one epoch of training, and returns current epoch accuracies."""
        tf.logging.info("training for iteration: {}".format(self._iteration))
        train_acc, val_acc = self.trainer.run_model(self._iteration)
        test_acc = self.trainer._compute_final_accuracies(self._iteration)
        return {"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc}

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint"""
        save_name = self.trainer.save_model(checkpoint_dir, self._iteration)
        tf.logging.info('saved model {}'.format(save_name))
        os.close(os.open(save_name, os.O_CREAT))
        return save_name

    def _restore(self, checkpoint):
        tf.logging.info("RESTORING: {}".format(checkpoint))
        self.trainer.extract_model_spec(checkpoint)


def main(_):
    from setup import create_parser, create_hparams
    FLAGS = create_parser("train")
    hparams = create_hparams("train", FLAGS)

    train_spec = {
        "run": RayModel,
        "trial_resources": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams.values(),
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    ray.init()
    run_experiments({FLAGS.name: train_spec})


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
