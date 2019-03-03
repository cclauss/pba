#!/bin/bash

train_reduced() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 7325 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --name reg_speedtest3 --gpu 0.19 --cpu 3 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/svhn_2_23_b_policy_15.txt" \
    --explore cifar10 --aug_policy cifar10 --hp_policy_epochs 160 \
    --lr 0.005 --wd 0.005 --num_samples 5 --no_cutout
}

search() {
    echo "[bash] Search on svhn"
    python search.py \
    --local_dir /data/dho/ray_results_2 \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 7325 --eval_test \
    --checkpoint_freq 0 \
    --name search_speedtest3 --gpu 0.19 --cpu 3 \
    --num_samples 16 --perturbation_interval 3 --epochs 160 \
    --explore cifar10 --aug_policy cifar10 --no_cutout \
    --lr 0.1 --wd 0.005
}

if [ "$1" = "reg" ]; then
    train_reduced "$@"
elif [ "$1" = "pba" ]; then
    search "$@"
else
    echo "invalid args"
    exit 1
fi
