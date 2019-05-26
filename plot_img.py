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

# load CIFAR
import torchvision
train_loader = torchvision.datasets.SVHN(
    root="/data/dho/datasets", split="train", download=True)
num_classes = 10
all_data = train_loader.data

all_data = all_data.reshape(-1, 3, 32, 32)
all_data = all_data.transpose(0, 2, 3, 1).copy()
all_data = all_data / 255.0
all_data = (all_data - augmentation_transforms_hp.MEANS["svhn_1000"]) / augmentation_transforms_hp.STDS["svhn_1000"]
np.random.seed(0)
perm = np.arange(len(all_data))
np.random.shuffle(perm)
all_data = all_data[perm]

img_no = 57
img_arr = all_data[img_no]

# for i in range(100):
#     img_preview = all_data[i]
#     img_preview = (img_preview * augmentation_transforms_hp.STDS['svhn_1000']) + augmentation_transforms_hp.MEANS['svhn_1000']
#     plt.imshow(img_preview)
#     plt.show()
#     plt.clf()
#     print(i)

# build policy
epochs = 160
all_schedule = parse_log_schedule('/home/danny/Documents/pba/schedules/svhn/svhn_2_23_b_policy_15.txt', epochs)
print(len(all_schedule))


for _ in range(10):
    final_imgs = [[img_arr, ''] for _ in range(4)]
    for i in range(2):
        for schedule in [all_schedule[19], all_schedule[59], all_schedule[99], all_schedule[159]]:
            # schedule = all_schedule[-1][1]
            split = len(schedule) // 2
            cur_pol = data_utils.parse_policy(
                schedule[:split], augmentation_transforms_hp)
            cur_pol.extend(data_utils.parse_policy(
                schedule[split:], augmentation_transforms_hp))
            # final_imgs.append([img_arr, '']) # (img, label)
            # for i in range(2):
            this_data, this_augs = augmentation_transforms_hp.policy_verbose(img_arr,i+1,cur_pol)
            final_imgs.append([this_data, this_augs])

    for i in range(len(final_imgs)):
        this_img = final_imgs[i][0]
        this_img = (this_img * augmentation_transforms_hp.STDS['svhn_1000']) + augmentation_transforms_hp.MEANS['svhn_1000']
        final_imgs[i][0] = this_img

    # plt stuff
    plt.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(3, 4, sharey=True, figsize=(15,20))
    plt.subplots_adjust(hspace=-0.25, wspace=0.1)
    fig.patch.set_visible(False)
    for ax in axes.flat:
        # ax.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    for i in range(4):
        axes[0][i].imshow(final_imgs[i][0])
        axes[0][i].set(xlabel=final_imgs[i][1])
    for i in range(4):
        axes[1][i].imshow(final_imgs[i+4][0])
        axes[1][i].set(xlabel=final_imgs[i+4][1])
    for i in range(4):
        axes[2][i].imshow(final_imgs[i+8][0])
        axes[2][i].set(xlabel=final_imgs[i+8][1])
    axes[0][0].set(ylabel='No Ops (20%)')
    axes[1][0].set(ylabel='1 Op (30%)')
    axes[2][0].set(ylabel='2 Ops (50%)')
    titles = ['Epoch 20','Epoch 60','Epoch 100','Epoch 160'] # ['No Ops (20%)', '1 Op (30%)', '2 Ops (50%)'] #, '3 Ops (10%)']
    for i in range(4):
        axes[0][i].set(title=titles[i])
        # f = ax.get_figure()
        # f.subplots_adjust(top=0.95)

    # for ax in axes.flat:
        # ax.label_outer()
    plt.show()
    # plt.clf()