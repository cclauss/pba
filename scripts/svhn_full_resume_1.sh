
# python train.py \
# --local_dir /data/dho/ray_results_2/svhn \
# --model_name wrn_28_10 --dataset svhn-full \
# --train_size 604388 --val_size 0 --eval_test \
# --checkpoint_freq 5 \
# --gpu 1 --cpu 4 \
# --use_hp_policy --hp_policy "/data/dho/pba/schedules/svhn/svhn_2_23_b_policy_15.txt" \
# --explore cifar10 \
# --hp_policy_epochs 160 --epochs 160 \
# --aug_policy cifar10 --name svhn_full_wrn2810 --num_samples 1 \
# --restore /home/danielho6868_gmail_com/checkpoint_40_c71_wrn/model.ckpt-40

python train.py \
--local_dir /data/dho/ray_results_2/svhn \
--model_name wrn_28_10 --dataset svhn-full \
--train_size 604388 --val_size 0 --eval_test \
--checkpoint_freq 5 \
--gpu 1 --cpu 4 \
--epochs 160 --name svhn_full_wrn2810_aa \
--lr 0.005 --wd 0.001
