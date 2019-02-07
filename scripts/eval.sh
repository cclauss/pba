#!/bin/bash
export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

# args: [model name] [number of times] [dataset name] [gpu portion]
eval() {
    echo "model: $1, trials: $2, dataset: $3"
    if [ "$3" = "r-cf10" ]; then
        size=4000
        name="reduced_cifar10_$1"
        dataset="cifar10"
        data_path="/data/dho/datasets/cifar-10-batches-py"
    elif [ "$3" = "cf10" ]; then
        size=50000
        name="cifar10_$1"
        dataset="cifar10"
        data_path="/data/dho/datasets/cifar-10-batches-py"
    elif [ "$3" == "cf100" ]; then
        size=50000
        name="cifar100_$1"
        dataset="cifar100"
        data_path="/data/dho/datasets/cifar-100-python"
    else
        echo "invalid dataset"
    fi

    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name "$1" \
    --data_path "$data_path" --dataset "$dataset" \
    --train_size "$size" --val_size 0 --eval_test \
    --checkpoint_freq 50 \
    --gpu "$4" --cpu 3 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy cifar10 --name "$name" --num_samples "$2"
}

# CUDA_VISIBLE_DEVICES=0 source ./scripts/eval.sh wrn_28_10 1 cf100 1 > eval_logs/cifar100_wrn_28_10.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 source ./scripts/eval.sh shake_shake_96 1 cf100 0.5 > eval_logs/cifar100_wrn_28_10.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 source ./scripts/eval.sh pyramidnet 1 cf100 1 > eval_logs/cifar100_wrn_28_10.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 ./scripts/eval.sh wrn_28_10 5 r-cf10 1 > eval_logs/rcf_wrn_28_10.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 source ./scripts/eval.sh shake_shake_96 1 r-cf10 0.5 > eval_logs/rcf10_ss96.txt 2>&1 &

# wrn shake-shake-32 shake-shake-96 shake-shake-112 pyramidnet
eval "$@"
