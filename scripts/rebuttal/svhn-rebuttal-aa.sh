#!/bin/bash

eval_svhn() {
    python train.py \
    --local_dir /data/dho/ray_results_2/svhn \
    --model_name shake_shake_96 --dataset "svhn-full" \
    --train_size 604388 --val_size 0 --eval_test \
    --checkpoint_freq 5 \
    --gpu 1 --cpu 4 \
    --epochs 160 --name "svhn_full_aa_$1" --num_samples 1
}

eval_svhn "$@"
