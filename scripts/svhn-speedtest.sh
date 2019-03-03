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
    --lr 0.05 --wd 0.05 --num_samples 5 --no_cutout
}

train_reduced_2() {
    python train.py \
    --local_dir /data/dho/ray_results_2 \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 46000 --eval_test \
    --checkpoint_freq 0 \
    --gpu 0.19 --cpu 3 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy cifar10 --name reg_cf10_speedtest3 --num_samples 5
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

search_2() {
    python search.py \
    --local_dir /data/dho/ray_results_2 \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 46000 --eval_test \
    --checkpoint_freq 0 \
    --name "cf10-speedtest3" --gpu 0.19 --cpu 3 \
    --num_samples 16 --perturbation_interval 3 --epochs 200 \
    --explore cifar10 --aug_policy cifar10 \
    --lr 0.1 --wd 0.0005
}

# if [ "$1" = "reg" ]; then
#     train_reduced "$@"
# elif [ "$1" = "pba" ]; then
#     search "$@"
# else
#     echo "invalid args"
#     exit 1
# fi

train_reduced_2
search_2
