#!/bin/bash
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=2

wrn_40_2_search() {
    echo "[bash] Search w/ wrn_40_2, policy ${1}"
    python search.py \
    --local_dir /data/dho/ray_results_2 \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 46000 --eval_test \
    --checkpoint_freq 0 \
    --name "search_aug-${1}" --gpu 0.15 --cpu 2 \
    --num_samples 16 --perturbation_interval 3 --epochs 200 \
    --explore cifar10 --aug_policy $1
}

if [ "$1" = "aug_11-23" ]; then
    echo "[bash] $@"
    wrn_40_2_search 11-23
else
    echo "invalid args"
fi
