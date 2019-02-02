#!/bin/bash
export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

wrn_40_2_eval() {
    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name wrn_40_2 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 4000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 0.19 --cpu 2 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/$3.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy "$1" --name "$3" --num_samples "$2"
}

# arguments: [$1 aug_policy] [$2 number of runs]
wrn_28_10_eval() {
    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name wrn_28_10 \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size 50000 --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/$3.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy "$1" --name "$3" --num_samples "$2"
}

#  number of trials wrn-40-2, number of trials wrn-28-10 schedule
# CUDA_VISIBLE_DEVICES=0 ./scripts/eval-wrn.sh 5 1 reduced_cifar_10/16_wrn.txt
# CUDA_VISIBLE_DEVICES=2,3,4 ./scripts/eval-wrn.sh 5 3 11-23_bair_policy_6 |& tee eval_logs/11-23_bair_policy_6.txt
# CUDA_VISIBLE_DEVICES=5,6,7 ./scripts/eval-wrn.sh 5 3 11-23_c66_policy_9 |& tee eval_logs/11-23_c66_policy_9.txt

# CUDA_VISIBLE_DEVICES=0 ./scripts/eval-wrn.sh 0 1 11-23_tuned_bair_policy_2 |& tee eval_logs/11-23_tuned_bair_policy_2.txt
# CUDA_VISIBLE_DEVICES=1 ./scripts/eval-wrn.sh 5 0 11-23_tuned_bair_policy_2 |& tee eval_logs/11-23_tuned_bair_policy_2_small.txt

# CUDA_VISIBLE_DEVICES=5,6,7 ./scripts/eval-wrn.sh 5 3 11-23_tuned_c66_policy_3 |& tee eval_logs/11-23_tuned_c66_policy_3.txt
# CUDA_VISIBLE_DEVICES=5,6,7 ./scripts/eval-wrn.sh 5 3 11-23_tuned_c69_policy_4 |& tee eval_logs/11-23_tuned_c69_policy_4.txt

wrn_28_10_eval 11-23 $2 $3
wrn_40_2_eval 11-23 $1 $3
