# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Data utils for CIFAR-10 and CIFAR-100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cPickle
import os
import numpy as np
import policies as found_policies
import tensorflow as tf
import random

from utils import parse_log_schedule


# pylint:disable=logging-format-interpolation

def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb) == 2 * num_xform, "policy was: {}, supposed to be: {}".format(
        len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2*i] / 10., policy_emb[2*i+1]))
    return policy


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0

        self.parse_policy(hparams)
        all_data, all_labels = self.load_data(hparams)
        # print(hparams.dataset, all_data.shape, type(all_data), all_data.dtype)

        # Break off test data
        if 'cifar' in hparams.dataset:
            train_dataset_size = 50000
            if hparams.eval_test:
                self.test_images = all_data[train_dataset_size:]
                self.test_labels = all_labels[train_dataset_size:]
                all_data = all_data[:train_dataset_size]
                all_labels = all_labels[:train_dataset_size]
            # Shuffle data for CIFAR only
            np.random.seed(0)
            perm = np.arange(len(all_data))
            np.random.shuffle(perm)
            all_data = all_data[perm]
            all_labels = all_labels[perm]
            # Break into train and val
            train_size, val_size = hparams.train_size, hparams.validation_size
            assert 50000 >= train_size + val_size
            self.train_images = all_data[:train_size]
            self.train_labels = all_labels[:train_size]
            self.val_images = all_data[train_size:train_size + val_size]
            self.val_labels = all_labels[train_size:train_size + val_size]
            self.num_train = self.train_images.shape[0]
        elif 'svhn' in hparams.dataset:
            assert hparams.eval_test
            test_dataset_size = 26032
            self.test_images = all_data[-test_dataset_size:]
            self.test_labels = all_labels[-test_dataset_size:]
            all_data = all_data[:-test_dataset_size]
            all_labels = all_labels[:test_dataset_size]
            train_size, val_size = hparams.train_size, hparams.validation_size
            if hparams.dataset == 'svhn-full':
                assert train_size + val_size <= 604388
            else:
                assert train_size + val_size <= 73257
            self.train_images = all_data[:train_size]
            self.train_labels = all_labels[:train_size]
            self.val_images = all_data[-val_size:]
            self.val_labels = all_labels[-val_size:]
            self.num_train = self.train_images.shape[0]

        # mean = self.train_images.mean(axis=(0,1,2))
        # std = self.train_images.std(axis=(0,1,2))
        # tf.logging.info('[train] mean:{}    std: {}'.format(mean, std))
        # if hparams.validation_size > 0:
        #     mean = self.val_images.mean(axis=(0,1,2))
        #     std = self.val_images.std(axis=(0,1,2))
        #     tf.logging.info('[eval] mean:{}    std: {}'.format(mean, std))
        # mean = self.test_images.mean(axis=(0,1,2))
        # std = self.test_images.std(axis=(0,1,2))
        # tf.logging.info('[test] mean:{}    std: {}'.format(mean, std))
        # exit()

        if hparams.eval_test:
            tf.logging.info("train dataset size: {}, test: {}, val: {}".format(
                train_size, len(self.test_images), val_size))
        else:
            tf.logging.info("train dataset size: {}, NO test, val: {}".format(
                train_size, val_size))


    def parse_policy(self, hparams):
        # Parse policy
        if hparams.use_hp_policy:
            import augmentation_transforms_hp as augmentation_transforms
            self.augmentation_transforms = augmentation_transforms

            if type(hparams.hp_policy) is str and hparams.hp_policy.endswith(".txt"):
                assert hparams.num_epochs % hparams.hp_policy_epochs == 0, (
                    hparams.num_epochs, hparams.hp_policy_epochs)
                tf.logging.info("schedule policy trained on {} epochs, parsing from: {}, multiplier: {}".format(
                    hparams.hp_policy_epochs, hparams.hp_policy, hparams.num_epochs//hparams.hp_policy_epochs))
                raw_policy = parse_log_schedule(
                    hparams.hp_policy, epochs=hparams.hp_policy_epochs, multiplier=hparams.num_epochs//hparams.hp_policy_epochs)
            elif type(hparams.hp_policy) is str and hparams.hp_policy.endswith(".p"):
                assert hparams.num_epochs % hparams.hp_policy_epochs == 0
                tf.logging.info(
                    "custom .p file, policy number: {}".format(hparams.schedule_num))
                with open(hparams.hp_policy, 'rb') as f:
                    policy = cPickle.load(f)[hparams.schedule_num]
                raw_policy = []
                for num_iters, pol in policy:
                    for _ in range(num_iters * hparams.num_epochs//hparams.hp_policy_epochs):
                        raw_policy.append(pol)
            else:
                raw_policy = hparams.hp_policy

            if type(raw_policy[0]) is list:
                self.policy = []
                split = len(raw_policy[0]) // 2
                for pol in raw_policy:
                    if 'svhn' in hparams.dataset:
                        cur_pol = parse_policy(
                            pol[:split], self.augmentation_transforms)
                    else:
                        cur_pol = parse_policy(
                            pol[:split], self.augmentation_transforms)
                        cur_pol.extend(parse_policy(
                            pol[split:], self.augmentation_transforms))
                    self.policy.append(cur_pol)
                tf.logging.info(
                    'using HP policy schedule, last: {}'.format(self.policy[-1]))
            elif type(raw_policy) is list:
                split = len(raw_policy) // 2
                if 'svhn' in hparams.dataset:
                    self.policy = parse_policy(
                        raw_policy, self.augmentation_transforms)
                else:
                    self.policy = parse_policy(
                        raw_policy[:split], self.augmentation_transforms)
                    self.policy.extend(parse_policy(
                        raw_policy[split:], self.augmentation_transforms))
                tf.logging.info(
                    "using HP Policy, policy: {}".format(self.policy))

        else:
            import augmentation_transforms
            self.augmentation_transforms = augmentation_transforms
            tf.logging.info("using ENAS Policy or no augmentaton policy")
            if 'svhn' in hparams.dataset:
                self.good_policies = found_policies.good_policies_svhn()
            else:
                assert 'cifar' in hparams.dataset
                self.good_policies = found_policies.good_policies()


    def load_data(self, hparams):
        all_labels = []
        if hparams.dataset == 'cifar10' or hparams.dataset == 'cifar100':
            num_data_batches_to_load = 5
            total_batches_to_load = num_data_batches_to_load
            train_batches_to_load = total_batches_to_load
            assert hparams.train_size + hparams.validation_size <= 50000
            if hparams.eval_test:
                total_batches_to_load += 1
            # Determine how many images we have loaded
            total_dataset_size = 10000 * num_data_batches_to_load
            if hparams.eval_test:
                total_dataset_size += 10000
            if hparams.dataset == 'cifar10':
                all_data = np.empty(
                    (total_batches_to_load, 10000, 3072), dtype=np.uint8)
                tf.logging.info('Cifar10')
                datafiles = [
                    'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                    'data_batch_5']

                datafiles = datafiles[:train_batches_to_load]
                if hparams.eval_test:
                    datafiles.append('test_batch')
                num_classes = 10
            elif hparams.dataset == 'cifar100':
                assert num_data_batches_to_load == 5
                all_data = np.empty((1, 50000, 3072), dtype=np.uint8)
                if hparams.eval_test:
                    test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
                datafiles = ['train']
                if hparams.eval_test:
                    datafiles.append('test')
                num_classes = 100
            for file_num, f in enumerate(datafiles):
                d = unpickle(os.path.join(hparams.data_path, f))
                if f == 'test':
                    test_data[0] = copy.deepcopy(d['data'])
                    all_data = np.concatenate([all_data, test_data], axis=1)
                else:
                    all_data[file_num] = copy.deepcopy(d['data'])
                if hparams.dataset == 'cifar10':
                    labels = np.array(d['labels'])
                else:
                    labels = np.array(d['fine_labels'])
                nsamples = len(labels)
                for idx in range(nsamples):
                    all_labels.append(labels[idx])
            all_data = all_data.reshape(total_dataset_size, 3072)
            all_data = all_data.reshape(-1, 3, 32, 32)
        elif hparams.dataset == 'svhn':
            assert hparams.train_size == 1000
            assert hparams.train_size + hparams.validation_size <= 73257
            import torchvision
            train_loader = torchvision.datasets.SVHN(
                root="/data/dho/datasets", split="train", download=True)
            test_loader = torchvision.datasets.SVHN(
                root="/data/dho/datasets", split="test", download=True)
            num_classes = 10
            all_data = np.concatenate(
                [train_loader.data, test_loader.data], axis=0)
            all_labels = np.concatenate(
                [train_loader.labels, test_loader.labels], axis=0)
        elif hparams.dataset == 'svhn-full':
            assert hparams.train_size == 73257 + 531131
            assert hparams.validation_size == 0
            import torchvision
            train_loader = torchvision.datasets.SVHN(
                root="/data/dho/datasets", split="train", download=True)
            test_loader = torchvision.datasets.SVHN(
                root="/data/dho/datasets", split="test", download=True)
            extra_loader = torchvision.datasets.SVHN(
                root="/data/dho/datasets", split="extra", download=True)
            num_classes = 10
            all_data = np.concatenate(
                [train_loader.data, extra_loader.data, test_loader.data], axis=0)
            all_labels = np.concatenate(
                [train_loader.labels, extra_loader.labels, test_loader.labels], axis=0)
            print(train_loader.data.shape,
                  test_loader.data.shape, extra_loader.data.shape)
        else:
            raise ValueError("unimplemented")
        all_data = all_data.transpose(0, 2, 3, 1).copy()
        all_data = all_data / 255.0
        mean = self.augmentation_transforms.MEANS[hparams.dataset +
                                             "_"+str(hparams.train_size)]
        std = self.augmentation_transforms.STDS[hparams.dataset +
                                           "_"+str(hparams.train_size)]
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        all_data = (all_data - mean) / std
        assert len(all_data) == len(all_labels)
        tf.logging.info(
            'In {} loader, number of images: {}'.format(hparams.dataset, len(all_data)))
        all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
        return all_data, all_labels


    def next_batch(self, iteration=None):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.hparams.batch_size
        if next_train_index > self.num_train:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:
                              self.curr_train_index + self.hparams.batch_size],
            self.train_labels[self.curr_train_index:
                              self.curr_train_index + self.hparams.batch_size])
        final_imgs = []

        dset = self.hparams.dataset + "_" + str(self.hparams.train_size)
        images, labels = batched_data
        for data in images:
            if not self.hparams.no_aug:
                if not self.hparams.use_hp_policy:
                    # apply autoaugment policy
                    epoch_policy = self.good_policies[np.random.choice(
                        len(self.good_policies))]
                    final_img = self.augmentation_transforms.apply_policy(
                        epoch_policy, data, dset=dset)
                else:
                    # apply PBA policy)
                    if type(self.policy[0]) is list:
                        # single policy
                        if self.hparams.flatten:
                            final_img = self.augmentation_transforms.apply_policy(
                                self.policy[random.randint(0, len(self.policy) - 1)], data, self.hparams.aug_policy, dset)
                        else:
                            final_img = self.augmentation_transforms.apply_policy(
                                self.policy[iteration], data, self.hparams.aug_policy, dset)
                    elif type(self.policy) is list:
                        # policy schedule
                        final_img = self.augmentation_transforms.apply_policy(
                            self.policy, data, self.hparams.aug_policy, dset)
                    else:
                        raise ValueError("unknown policy")
            else:
                # no extra
                final_img = data
            if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'cifar100':
                final_img = self.augmentation_transforms.random_flip(
                    self.augmentation_transforms.zero_pad_and_crop(final_img, 4))
            else:
                assert "svhn" in self.hparams.dataset
            # Apply cutout
            if not self.hparams.no_cutout:
                final_img = self.augmentation_transforms.cutout_numpy(final_img)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        self.curr_train_index += self.hparams.batch_size
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        assert self.num_train == self.train_images.shape[
            0], 'Error incorrect shuffling mask'
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        self.curr_train_index = 0


def unpickle(f):
    tf.logging.info('loading file: {}'.format(f))
    fo = tf.gfile.Open(f, 'r')
    d = cPickle.load(fo)
    fo.close()
    return d
