#!/bin/bash
# export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

# ./scripts/svhn.sh eval_svhn shake_shake_96 5 r-svhn svhn_2_23_b_policy_15
# ./scripts/svhn.sh eval_svhn shake_shake_96 5 r-svhn svhn_2_23_d_policy_11


# ./scripts/svhn.sh eval_svhn wrn_28_10 1 svhn-full
# ./scripts/svhn.sh eval_svhn shake_shake_96 1 svhn-full |& tee svhn_full_ss96.txt

# args: [] [model name] [number of times] [dataset name] [policy name]
eval_svhn() {
    echo "model: $2, trials: $3"
    if [ "$4" = "r-svhn" ]; then
        size=1000
        name="reduced_svhn_$2_$4"
        dataset="svhn"
    elif [ "$4" = "svhn-full" ]; then
        size=604388
        name="svhn_$2_$4"
        dataset="svhn-full"
    else
        echo "invalid dataset"
    fi

    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name "$2" --dataset "$dataset" \
    --train_size "$size" --val_size 0 --eval_test \
    --checkpoint_freq 5 \
    --gpu 1 --cpu 8 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/svhn_2_23_b_policy_15.txt" \
    --explore cifar10 \
    --hp_policy_epochs 160 --epochs 160 \
    --aug_policy cifar10 --name "$name" --num_samples "$3"

    # using default lr / wd
}

# train_clean() {
#     python train.py \
#     --local_dir /data/dho/ray_results_2/svhn \
#     --model_name wrn_40_2 --dataset svhn-full \
#     --train_size 604388 --val_size 0 --eval_test \
#     --checkpoint_freq 10 --epochs 160 \
#     --name svhn_full_clean --gpu 1 --cpu 6 \
#     --lr 0.005 --wd 0.001 --bs 128
# }

# train_aa() {
#     python train.py \
#     --local_dir /data/dho/ray_results_2/svhn \
#     --model_name wrn --dataset svhn-full \
#     --train_size 604388 --val_size 0 --eval_test \
#     --checkpoint_freq 10 --epochs 160 \
#     --name svhn_full_autoaug_wrn --gpu 1 --cpu 2 \
#     --lr 0.005 --wd 0.001 --bs 128
# }

# train() {
#     python train.py \
#     --local_dir /data/dho/ray_results_2/svhn \
#     --model_name wrn_40_2 --dataset svhn-full \
#     --train_size 604388 --val_size 0 --eval_test \
#     --checkpoint_freq 10 --epochs 160 \
#     --name svhn_full_autoaug_wrn_cifarpol --gpu 1 --cpu 2 \
#     --use_hp_policy --hp_policy "/data/dho/fast-hp-search/experiments/autoaugment/ray/schedules/policy_6-lowmag-wrn-16-200ep-4k.txt" \
#     --hp_policy_epochs 200 \
#     --policy_type double --param_type fixed_magnitude \
#     # --lr 0.005 --wd 0.001 --bs 128
# }

train_reduced_clean() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name wrn_28_10 --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 --epochs 160 \
    --name sanity --gpu 1 --cpu 2 --no_cutout --no_aug \
    --lr 0.1 --wd 0.0075 --bs 128
    # --lr 0.025 --bs 8
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
    --name sanity_aa --gpu 1 --cpu 2 --no_cutout \
    --lr 0.05 --wd 0.005 --bs 128
}

# ./scripts/svhn.sh train-reduced reduced_cifar_10/16_wrn_160 rsvhn_cifarpol_wrn_2810 wrn_28_10 0.05 0.05 160 |& tee rsvhn_cifarpol_wrn2810.txt
# ./scripts/svhn.sh train-reduced svhn/svhn_2_23_b_policy_15 rsvhn_svhnpol_ss96 shake_shake_96 0.005 0.005 160 |& tee rsvhn_svhnpol_ss96.txt
# ./scripts/svhn.sh train-reduced svhn/svhn_2_23_b_policy_15 rsvhn_svhnpol_ss96 shake_shake_96 0.005 0.005 1760 |& tee rsvhn_svhnpol_ss96.txt

# [] policy name name model_name lr wd epochs
train_reduced() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name $4 --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 --epochs $7 \
    --name $3 --gpu 1 --cpu 6 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/$2.txt" \
    --explore cifar10 --aug_policy cifar10 --hp_policy_epochs 160 \
    --lr $5 --wd $6 --num_samples 5
}

# name aug_policy lr
search() {
    echo "[bash] Search on svhn"
    python search.py \
    --local_dir /data/dho/ray_results_2 \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 7325 --eval_test \
    --checkpoint_freq 0 \
    --name "$2" --gpu 0.19 --cpu 2 \
    --num_samples 16 --perturbation_interval 3 --epochs 160 \
    --explore cifar10 --aug_policy "$3" --no_cutout \
    --lr "$4" --wd "$5"
}

# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_a cifar10 0.10 |& tee svhn_search_2_20_a
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_b cifar10 0.025 |& tee svhn_search_2_20_b
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_c 1-17-a 0.05 |& tee svhn_search_2_20_c
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_d 1-18-a 0.05 |& tee svhn_search_2_20_d
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_e 1-18-b 0.05 0.0005 |& tee svhn_search_2_20_e
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_f 11-23 0.05 0.0005 |& tee svhn_search_2_20_f
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_i cifar10 0.05 0.01 |& tee svhn_search_2_20_i
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_j cifar10 0.05 0.005 |& tee svhn_search_2_20_j

# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_g2 cifar10 0.05 0.00005 |& tee svhn_search_2_20_g2
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh search svhn_search_2_20_h2 cifar10 0.05 0.02 |& tee svhn_search_2_20_h2
# ./scripts/svhn.sh train-reduced

if [ "$1" = "train-reduced" ]; then
    echo "[bash] train-reduced $@"
    train_reduced "$@"
elif [ "$1" = "train-full" ]; then
    echo "[bash] train-full $@"
    exit 1
elif [ "$1" = "train-full-clean" ]; then
    echo "[bash] train-full $@"
    train_clean
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
    search "$@"
elif [ "$1" = "eval_svhn" ]; then
    echo "[bash] eval $@"
    eval_svhn "$@"
else
    echo "invalid args"
    exit 1
fi
