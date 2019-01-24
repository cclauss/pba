<font size=4><b>Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules</b></font>

Started from public code from AutoAugment authors which reproduce evaluation results on CIFAR 10 models, located at: https://github.com/tensorflow/models/tree/master/research/autoaugment

The CIFAR-10/CIFAR-100 data can be downloaded from:
https://www.cs.toronto.edu/~kriz/cifar.html.

This Python 2 code reproduces results in the PBA manuscript, including running the PBA search algorithm and evaluating performance on benchmark datasets. It also allows for evaluation of the AutoAugment policy using our training pipeline and hyperparameters.

<b>Settings:</b>

CIFAR-10&100 Model         | Learning Rate | Weight Decay | Num. Epochs | Batch Size
---------------------- | ------------- | ------------ | ----------- | ----------
Wide-ResNet-28-10      | 0.1           | 5e-4         | 200         | 128
Shake-Shake (26 2x32d) | 0.01          | 1e-3         | 1800        | 128
Shake-Shake (26 2x96d) | 0.01          | 1e-3         | 1800        | 128
PyramidNet + ShakeDrop | 0.05          | 5e-5         | 1800        | 64

Note: For CIFAR-100 & Reduced CIFAR 10, better performance can most likely be achived by tuning hyperparameters from this original set.

<b>Prerequisite:</b>

1.  Install requirements:

```shell
pip install -r requirements.txt
```

2.  Download CIFAR-10/CIFAR-100 dataset.

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

<b>Overview:</b>
We summarize the files in this repository below.

PBA:
- `train.py`: Evaluate model performance using Ray framework
- `search.py`: Runs PBA search using Ray
    - contains Algorithm 2 from manuscript, `explore()`
- `setup.py`: Flags and training parameters
- `utils.py`: Helper functions for PBA augmentation policy formulation
- `resnet.py`: Code for Resnet-20 child model
- `schedules/`: contains PBA augmentation schedules
- `schedules/`: contains PBA schedules from manuscript.
    -`schedules/reduced_cifar_10/16_wrn.txt` contains the schedule for Table 1.

Modified from AutoAugment:
- `augmentation_transforms_hp`: Copied from `augmentation_transforms.py`, modified for PBA augmentation policy formulation
    - contains Algorithm 1 from manuscript, `apply_policy()`
- `data_utils.py`: Dataloading, modified for PBA augmentation policy formulation
- `helper_utils.py`: Commonly used TensorFLow code
- `wrn.py`: WRN-28-10 code, added WRN-40-2
- `train_cifar.py`: Builds the model object, made compatible with Ray and PBA

Unmodified from AutoAugment:
- `augmentation_transforms`: Implementation of augmentation operations
- `custom_ops.py`: Helper TensorFlow ops
- `policies.py`: Hardcoded AutoAugment augmentation policy trained on Reduced CIFAR 10
- `shake_drop.py`: Implementation of PyramidNet+ShakeDrop network
- `shake_shake.py`: Implemenation of Shake-Shake networks

<b>Run Search Experiments</b>
Run PBA search with the following command. You can vary the child model and population size through flags. You can specify a partial GPU size to launch multiple trials on the same GPU.

```shell
python search.py \
 --local_dir [where to log results] \
 --model_name [wrn, resnet, shake_shake_32, pyramidnet, ...] --resnet_size 32 --wrn_depth 40 \
 --data_path /tmp/datasets/cifar-10-batches-py --dataset cifar10 \
 --train_size 4000 --val_size 45000 --eval_test \
 --name search --gpu 0.2 --cpu 3 \
 --num_samples 16 --perturbation_interval 3 --epochs 200
```

<b>Run Evaluation</b>
Run evaluation of PBA schedules and re-evaluate the AutoAugment policy using our hyperparameters.

Specify a PBA schedule through the `--use_hp_policy --hp_policy [path]` flags, or don't specify `use_hp_policy` to re-evaluate AutoAugment.

Random search scheules are stored in `schedules/ablations/`.

```shell
# For WRN-28-10, use --resnet_size 160 --wrn_depth 28
python train.py \
 --local_dir [where to log results] \
 --model_name [wrn, resnet, ...] --resnet_size 32 --wrn_depth 40 \
 --data_path /tmp/datasets/cifar-10-batches-py --dataset cifar10 \
 --train_size 50000 --val_size 0 --eval_test \
 --name train --gpu 1 --cpu 3 \
 --use_hp_policy --hp_policy "[full_path_to_repo]/schedules/reduced_cifar_10/16_wrn.txt" --hp_policy_epochs 200
```
