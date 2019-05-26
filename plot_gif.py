import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import augmentation_transforms_hp
import data_utils
import os
import cPickle
from data_utils import unpickle
import copy
from utils import parse_log_schedule
import seaborn
import matplotlib.gridspec as gridspec

# load svhn
import torchvision
import imageio

train_loader = torchvision.datasets.SVHN(
    root="/data/dho/datasets", split="train", download=True)
num_classes = 10
all_data = train_loader.data
all_data = all_data.reshape(-1, 3, 32, 32)
all_data = all_data[:50,:,:,:]
all_data = all_data.transpose(0, 2, 3, 1).copy()
all_data = all_data / 255.0
all_data = (all_data - augmentation_transforms_hp.MEANS["svhn_1000"]) / augmentation_transforms_hp.STDS["svhn_1000"]
np.random.seed(0)
perm = np.arange(len(all_data))
np.random.shuffle(perm)
all_data = all_data[perm]
all_schedule = parse_log_schedule('/home/danny/Documents/pba/schedules/svhn/svhn_2_23_b_policy_15.txt', 160)

final_svhn_imgs = []
for sch_idx in list(range(0, 160, 16)) + [159]:
    sch_imgs = []
    for img_idx in range(8):
        img_arr = all_data[img_idx]
        schedule = all_schedule[sch_idx]
        split = len(schedule) // 2
        cur_pol = data_utils.parse_policy(
            schedule[:split], augmentation_transforms_hp)
        cur_pol.extend(data_utils.parse_policy(
            schedule[split:], augmentation_transforms_hp))
        this_img = augmentation_transforms_hp.apply_policy(cur_pol, img_arr, 'cifar10', 'svhn_1000')
        this_img = (this_img * augmentation_transforms_hp.STDS['svhn_1000']) + augmentation_transforms_hp.MEANS['svhn_1000']
        sch_imgs.append(this_img)
    final_svhn_imgs.append(sch_imgs)


# load cifar
d = unpickle('/home/danny/.data/cifar-10-batches-py/data_batch_1')
all_data = d['data']
all_data = all_data.reshape(10000, 3072)
all_data = all_data[:50,:]
all_data = all_data.reshape(-1, 3, 32, 32)
all_data = all_data.transpose(0, 2, 3, 1).copy()
all_data = all_data / 255.0
mean = augmentation_transforms_hp.MEANS["cifar10_4000"]
std = augmentation_transforms_hp.STDS["cifar10_4000"]
all_data = (all_data - mean) / std
np.random.seed(0)
perm = np.arange(len(all_data))
np.random.shuffle(perm)
all_data = all_data[perm]
all_schedule = parse_log_schedule('/home/danny/Documents/pba/schedules/reduced_cifar_10/16_wrn.txt', 200)
final_cifar_imgs = []
for sch_idx in list(range(0, 200, 20)) + [199]:
    sch_imgs = []
    for img_idx in range(8):
        img_arr = all_data[img_idx]
        schedule = all_schedule[sch_idx]
        split = len(schedule) // 2
        cur_pol = data_utils.parse_policy(
            schedule[:split], augmentation_transforms_hp)
        cur_pol.extend(data_utils.parse_policy(
            schedule[split:], augmentation_transforms_hp))
        this_img = augmentation_transforms_hp.apply_policy(cur_pol, img_arr, 'cifar10', 'cifar10_4000')
        this_img = (this_img * augmentation_transforms_hp.STDS['cifar10_4000']) + augmentation_transforms_hp.MEANS['cifar10_4000']
        sch_imgs.append(this_img)
    final_cifar_imgs.append(sch_imgs)

print(len(final_svhn_imgs), len(final_cifar_imgs))
final_imgs = []
for i in range(len(final_svhn_imgs)):
    final_imgs.append(final_svhn_imgs[i] + final_cifar_imgs[i])

rendered = []
# plt stuff
for plot_img in range(len(final_imgs)):
    matplotlib.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(4, 4, figsize=(20,20))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.patch.set_visible(False)
    fig.suptitle('{}%'.format(plot_img * 10))
    for ax in axes.flat:
        # ax.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    for i in range(4):
        for j in range(4):
            axes[i][j].imshow(final_imgs[plot_img][i*4+j])
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    rendered.append(image)

imageio.mimsave('./test.gif', rendered, fps=1)
