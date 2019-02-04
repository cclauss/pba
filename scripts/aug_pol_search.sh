#!/bin/bash
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=2

# FLAGS:
# [$1 run name] [$2 number of runs]

# example:
# CUDA_VISIBLE_DEVICES=0 ./scripts/aug_pol_search.sh sm-2 2>&1 |tee aug_search-2.txt
# CUDA_VISIBLE_DEVICES=0 ./scripts/aug_pol_search.sh lg-1 2>&1 |tee aug_search_lg-1.txt
# CUDA_VISIBLE_DEVICES=0 ./scripts/aug_pol_search.sh lg-2 2 > aug_search_lg-2-pt2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 ./scripts/aug_pol_search.sh lg-2 1 > aug_search_lg-2-pt3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 ./scripts/aug_pol_search.sh lg-4 2 > aug_search_lg-4-pt2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=6 ./scripts/aug_pol_search.sh lg-4 1 > aug_search_lg-4-pt3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 ./scripts/aug_pol_search.sh lg-7 2 > aug_search_lg-7-pt2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=7 ./scripts/aug_pol_search.sh lg-7 1 > aug_search_lg-7-pt3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 ./scripts/aug_pol_search.sh lg-8 2 > aug_search_lg-8-pt2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 ./scripts/aug_pol_search.sh lg-8 1 > aug_search_lg-8-pt3.txt 2>&1 &

# arguments: [$1 aug_policy] [$2 number of runs]
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

# arguments: [$1 aug_policy] [$2 number of runs]
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

if [ "$1" = "sm-1" ]; then
    echo "[bash] $@"
    wrn_40_2_eval cifar10 $2
    wrn_40_2_eval 11-23 $2
elif [ "$1" = "sm-2" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 11-26 $2
    wrn_40_2_eval 11-29-a $2
elif [ "$1" = "sm-3" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-24-a $2
    wrn_40_2_eval 12-24-b $2
elif [ "$1" = "sm-4" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-24-c $2
    wrn_40_2_eval 12-24-d $2
elif [ "$1" = "sm-5" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-26-a $2
    wrn_40_2_eval 12-26-b $2
elif [ "$1" = "sm-6" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 12-26-c $2
    wrn_40_2_eval 1-15-a $2
elif [ "$1" = "sm-7" ]; then
    echo "[bash] $@"
    wrn_40_2_eval 11-28-a $2
    wrn_40_2_eval 11-28-b $2
elif [ "$1" = "sm-sanity" ]; then
    echo "[bash] $@"
    wrn_40_2_eval sanity $2
elif [ "$1" = "sm-sanity2" ]; then
    echo "[bash] $@"
    wrn_40_2_eval sanity-2 $2


elif [ "$1" = "lg-1" ]; then
    echo "[bash] $@"
    wrn_28_10_eval cifar10 $2
elif [ "$1" = "lg-2" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-23 $2
elif [ "$1" = "lg-3" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-26 $2
elif [ "$1" = "lg-4" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-29-a $2
elif [ "$1" = "lg-5" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-a $2
elif [ "$1" = "lg-6" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-b $2
elif [ "$1" = "lg-7" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-c $2
elif [ "$1" = "lg-8" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-24-d $2
elif [ "$1" = "lg-9" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-a $2
elif [ "$1" = "lg-10" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-b $2
elif [ "$1" = "lg-11" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 12-26-c $2
elif [ "$1" = "lg-12" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 1-15-a $2
elif [ "$1" = "lg-13" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-28-a $2
elif [ "$1" = "lg-14" ]; then
    echo "[bash] $@"
    wrn_28_10_eval 11-28-b $2

else
    echo "invalid args"
fi
