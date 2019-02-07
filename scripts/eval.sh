#!/bin/bash
export PYTHONPATH="$(pwd)"
# export OMP_NUM_THREADS=2

# args: [model name] [number of times] [dataset name]
eval() {
    echo "$1 $2 $3"
    if [ "$3" = "r-cf10" ]; then
        size=4000
        name="reduced_cifar10_$1"
    elif [ "$3" = "cf10" ]; then
        size=50000
        name="cifar10_$1"
    else
        echo "invalid dataset"
    fi

    python train.py \
    --local_dir /data/dho/ray_results_2/eval \
    --model_name "$1" \
    --data_path /data/dho/datasets/cifar-10-batches-py --dataset cifar10 \
    --train_size "$size" --val_size 0 --eval_test \
    --checkpoint_freq 0 \
    --gpu 1 --cpu 3 \
    --use_hp_policy --hp_policy "/data/dho/pba/schedules/reduced_cifar_10/16_wrn.txt" \
    --explore cifar10 \
    --hp_policy_epochs 200 \
    --aug_policy cifar10 --name "$name" --num_samples "$2" > "eval_logs/$name.txt" 2>&1 &
}

# CUDA_VISIBLE_DEVICES=0 source ./scripts/eval-wrn.sh 5 1  reduced-cifar10
# wrn shake-shake-32 shake-shake-96 shake-shake-112 pyramidnet
eval $1 $2 $3
