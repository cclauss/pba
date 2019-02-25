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

svhn_wrn_40_2_grid_search() {
    echo "[bash] SVHN Grid Search w/ wrn_40_2, policy ${1}"
    python grid_search.py \
    --local_dir /data/dho/ray_results_2/svhn_grid_search \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 0.166 --cpu 1 \
    --explore cifar10 --no_cutout --name "svhn_gs"
}

svhn_pba_wrn_40_2_grid_search() {
    echo "[bash] SVHN Grid Search w/ wrn_40_2, policy ${1}"
    python grid_search.py \
    --local_dir /data/dho/ray_results_2/svhn_grid_search \
    --model_name wrn_40_2 --dataset svhn \
    --gpu 0.166 --cpu 1 \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --explore cifar10 --no_cutout --name "pba_gs_$2_wrn402" \
    --hp_policy_epochs 160 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/$2.txt" \
}

svhn_pba_wrn_28_10_grid_search() {
    echo "[bash] SVHN Grid Search w/ wrn_28_10, policy ${1}"
    python grid_search.py \
    --local_dir /data/dho/ray_results_2/svhn_grid_search \
    --model_name wrn_28_10 --dataset svhn \
    --gpu 1 --cpu 8 \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --explore cifar10 --no_cutout --name "pba_gs_$2_wrn2810" \
    --hp_policy_epochs 160 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/$2.txt" \
}

svhn_pba_ss_96_grid_search() {
    echo "[bash] SVHN Grid Search w/ ss96, policy ${1}"
    python grid_search.py \
    --local_dir /data/dho/ray_results_2/svhn_grid_search \
    --model_name shake_shake_96 --dataset svhn \
    --gpu 0.5 --cpu 4 \
    --train_size 1000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --explore cifar10 --no_cutout --name "pba_gs_$2_ss96" \
    --hp_policy_epochs 160 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/$2.txt" \
}

# ./scripts/grid_search.sh svhn_pba_wrn_40_2 svhn_2_23_b_policy_15
# ./scripts/grid_search.sh svhn_pba_wrn_40_2 svhn_2_23_d_policy_11
# ./scripts/grid_search.sh svhn_pba_wrn_28_10 svhn_2_23_b_policy_15
# ./scripts/grid_search.sh svhn_pba_wrn_28_10 svhn_2_23_d_policy_11
# ./scripts/grid_search.sh svhn_pba_ss_96 svhn_2_23_b_policy_15
# ./scripts/grid_search.sh svhn_pba_ss_96 svhn_2_23_d_policy_11

if [ "$1" = "aug_11-23" ]; then
    echo "[bash] $@"
    wrn_40_2_grid_search 11-23
elif [ "$1" = "svhn" ]; then
    echo "[bash] $@"
    svhn_wrn_40_2_grid_search
elif [ "$1" = "svhn_pba_wrn_40_2" ]; then
    echo "[bash] $@"
    svhn_pba_wrn_40_2_grid_search "$@"
elif [ "$1" = "svhn_pba_wrn_28_10" ]; then
    echo "[bash] $@"
    svhn_pba_wrn_28_10_grid_search "$@"
elif [ "$1" = "svhn_pba_ss_96" ]; then
    echo "[bash] $@"
    svhn_pba_ss_96_grid_search "$@"
else
    echo "invalid args"
fi
