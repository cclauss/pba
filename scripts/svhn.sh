#!/bin/bash
# export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

train_aa() {
    python train.py \
    --model_name wrn --dataset svhn_extra \
    --train_size 604388 --val_size 0 --eval_test \
    --checkpoint_freq 10 --epochs 160 \
    --name svhn_full_autoaug_wrn --gpu 1 --cpu 2 \
    --lr 0.005 --wd 0.001 --bs 128
}

train() {
    python train.py \
    --model_name wrn --dataset svhn_extra \
    --train_size 604388 --val_size 0 --eval_test \
    --checkpoint_freq 10 --epochs 160 \
    --name svhn_full_autoaug_wrn_cifarpol --gpu 1 --cpu 2 \
    --use_hp_policy --hp_policy "/data/dho/fast-hp-search/experiments/autoaugment/ray/schedules/policy_6-lowmag-wrn-16-200ep-4k.txt" \
    --hp_policy_epochs 200 \
    --policy_type double --param_type fixed_magnitude \
    # --lr 0.005 --wd 0.001 --bs 128
}

train_reduced_clean() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --name sanity --gpu 1 --cpu 2 --no_cutout --no_aug \
    --lr 0.025 --bs 8
    # --lr 0.1 --wd 0.0075 --bs 128
}


train_reduced_aa() {
    # echo "model: $1, trials: $2, dataset: $3"
    # if [ "$3" = "r-cf10" ]; then
    #     size=4000
    #     name="reduced_cifar10_$1"
    #     dataset="cifar10"
    #     data_path="/data/dho/datasets/cifar-10-batches-py"
    # elif [ "$3" = "cf10" ]; then
    #     size=50000
    #     name="cifar10_$1"
    #     dataset="cifar10"
    #     data_path="/data/dho/datasets/cifar-10-batches-py"
    # elif [ "$3" == "cf100" ]; then
    #     size=50000
    #     name="cifar100_$1"
    #     dataset="cifar100"
    #     data_path="/data/dho/datasets/cifar-100-python"
    # else
    #     echo "invalid dataset"
    # fi

    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --name sanity --gpu 1 --cpu 2
    # --lr 0.005 --wd 0.005 --bs 128
}

train_reduced() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name wrn --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --name svhn_1-16-a-eval --gpu 1 --cpu 2 \
    --use_hp_policy --hp_policy "/data/dho/fast-hp-search/experiments/autoaugment/ray/schedules/svhn/policy_2-1-16-a.txt" \
    --hp_policy_epochs 160 \
    --explore 12-4-b-mod --aug_policy 11-26 --policy_type single --param_type fixed_magnitude \
    # --lr 0.005 --wd 0.005 --bs 128
    # --lr 0.025 --bs 8
}

search() {
    python search.py \
    --model_name wrn --dataset svhn \
    --train_size 1000 --val_size 7000 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --num_samples 16 --perturbation_interval 3 \
    --name svhn_1-18-a --gpu 0.16 --cpu 2 \
    --no_cutout --lr 0.025 --bs 8 \
    --explore 1-2-mod-bias --aug_policy 1-18-a --policy_type single --param_type fixed_magnitude
}

if [ "$1" = "train-reduced" ]; then
    echo "[bash] train-reduced $@"
    exit 1
elif [ "$1" = "train-full" ]; then
    echo "[bash] train-full $@"
    exit 1
elif [ "$1" = "train-reduced-aa" ]; then
    echo "[bash] train-reduced-aa $@"
    train_reduced_aa
elif [ "$1" = "train-reduced-clean" ]; then
    echo "[bash] train-reduced-aa $@"
    train_reduced_clean
elif [ "$1" = "train-full-aa" ]; then
    echo "[bash] train-full-aa $@"
    exit 1
elif [ "$1" = "search" ]; then
    echo "[bash] search $@"
    exit 1
else
    echo "invalid args"
    exit 1
fi
