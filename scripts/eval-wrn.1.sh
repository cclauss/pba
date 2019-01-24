#!/bin/bash
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=2

python train.py \
 --local_dir /data/dho/ray_results \
 --model_name wrn --resnet_size 160 --wrn_depth 28 \
 --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
 --train_size 50000 --val_size 0 --eval_test \
 --name train --gpu 1 --cpu 3 \
 --use_hp_policy --hp_policy "/data/dho/ICML_sub/experiments/autoaugment/PBA/schedules/reduced_cifar_10/16_wrn.txt" \
 --hp_policy_epochs 200 --aug_policy 11-29-a 
