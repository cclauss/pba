#!/bin/bash
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=2

# example:
# CUDA_VISIBLE_DEVICES=0 ./scripts/aug_pol_search.sh sm-2 2>&1 |tee aug_search-2.txt

wrn_40_2_eval() {
    for i in $(seq 1 $2)
    do
        echo "[bash] Run wrn_40_2, iter $i, with policy $1"
        python train.py \
        --local_dir /data/dho/ray_results_2 \
        --model_name wrn_40_2 \
        --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
        --train_size 4000 --val_size 0 --eval_test \
        --checkpoint_freq 0 \
        --gpu 1 --cpu 3 \
        --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
        --explore cifar10 \
        --hp_policy_epochs 200 \
        --aug_policy "$1" --name "aug_policy-$1"
    done
}

wrn_28_10_eval() {
    for i in $(seq 1 $2)
    do
        echo "[bash] Run wrn_28_10, iter $i, with policy $1"
        python train.py \
        --local_dir /data/dho/ray_results_2 \
        --model_name wrn_28_10 \
        --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
        --train_size 50000 --val_size 0 --eval_test \
        --checkpoint_freq 0 \
        --gpu 1 --cpu 3 \
        --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
        --explore cifar10 \
        --hp_policy_epochs 200 \
        --aug_policy "$1" --name "aug_policy-$1"
    done
}

if [ "$@" = "sm-1" ]; then
    echo "[bash] $@"
    wrn_40_2_eval cifar10 5
    wrn_40_2_eval 11-23 5
elif [ "$@" = "sm-2" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 11-26 5
    wrn_40_2_eval 11-29-a 5
elif [ "$@" = "sm-3" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-24-a 5
    wrn_40_2_eval 12-24-b 5
elif [ "$@" = "sm-4" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-24-c 5
    wrn_40_2_eval 12-24-d 5
elif [ "$@" = "sm-5" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-26-a 5
    wrn_40_2_eval 12-26-b 5
elif [ "$@" = "sm-6" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-26-c 5
    wrn_40_2_eval 1-15-a 5
elif [ "$@" = "sm-7" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 11-28-a 5
    wrn_40_2_eval 11-28-b 5

elif [ "$@" = "lg-1" ]; then
    echo "[bash] $@"
    wrn_28_10_eval cifar10 2
elif [ "$@" = "lg-2" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-23 2
elif [ "$@" = "lg-3" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-26 2
elif [ "$@" = "lg-4" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-29-a 2
elif [ "$@" = "lg-5" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-a 2
elif [ "$@" = "lg-6" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-b 2
elif [ "$@" = "lg-7" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-c 2
elif [ "$@" = "lg-8" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-d 2
elif [ "$@" = "lg-9" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-a 2
elif [ "$@" = "lg-10" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-b 2
elif [ "$@" = "lg-11" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-c 2
elif [ "$@" = "lg-12" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 1-15-a 2
elif [ "$@" = "lg-13" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-28-a 2
elif [ "$@" = "lg-14" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-28-b 2

else
    echo "invalid args"
fi