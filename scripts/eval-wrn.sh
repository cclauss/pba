#!/bin/bash
export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

wrn_40_2_eval() {
    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 0.142 --cpu 2 \
    --use_hp_policy --hp_policy "/data/dho/pba/$3" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy "$1" --name "$3" --num_samples "$2"
}

# arguments: [$1 aug_policy] [$2 number of runs]
wrn_28_10_eval() {
    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name wrn_28_10 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 50000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "/data/dho/pba/$3" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy "$1" --name "$3" --num_samples "$2"
}

#  number of trials wrn-40-2, number of trials wrn-28-10 schedule
# CUDA_VISIBLE_DEVICES=0 source ./eval-wrn.sh 5 1 /data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt

wrn_28_10_eval 11-23 $2 $3
wrn_40_2_eval 11-23 $1 $3
