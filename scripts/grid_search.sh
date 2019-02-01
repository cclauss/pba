#!/bin/bash
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=2

# CUDA_VISIBLE_DEVICES=0 ./scripts/search.sh aug_11-23 |& tee search_aug_11-23-pt1.txt

wrn_40_2_grid_search() {
    echo "[bash] Grid Search w/ wrn_40_2, policy ${1}"
    python grid_search.py \
    --local_dir /data/dho/ray_results_2/grid_search_3 \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 0.24 --cpu 1 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy "$1" --name "aug_policy-$1"
}

if [ "$1" = "aug_11-23" ]; then
    echo "[bash] $@"
    wrn_40_2_grid_search 11-23
else
    echo "invalid args"
fi
