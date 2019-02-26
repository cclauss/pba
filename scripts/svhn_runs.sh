# CUDA_VISIBLE_DEVICES=0 ./scripts/grid_search.sh svhn_pba_wrn_40_2 svhn_2_23_b_policy_15 |& tee gs_wrn_40_2_svhn_2_23_b_policy_15.txt
# CUDA_VISIBLE_DEVICES=0 ./scripts/grid_search.sh svhn_pba_wrn_28_10 svhn_2_23_b_policy_15 |& tee gs_wrn_28_10_svhn_2_23_b_policy_15.txt
# CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh eval_svhn shake_shake_96 5 r-svhn svhn_2_23_b_policy_15 |& tee eval_r_svhn_ss96_2_23_b_policy_15.txt


#CUDA_VISIBLE_DEVICES=0 ./scripts/grid_search.sh svhn_pba_wrn_40_2 svhn_2_23_d_policy_11 |& tee gs_wrn_40_2_svhn_2_23_d_policy_11.txt
#CUDA_VISIBLE_DEVICES=0 ./scripts/grid_search.sh svhn_pba_wrn_28_10 svhn_2_23_d_policy_11 |& tee gs_wrn_28_10_svhn_2_23_d_policy_11.txt
#CUDA_VISIBLE_DEVICES=0 ./scripts/svhn.sh eval_svhn shake_shake_96 5 r-svhn svhn_2_23_d_policy_11 |& tee eval_r_svhn_2_23_d_policy_11.txt

./scripts/svhn.sh train-reduced reduced_cifar_10/16_wrn_160 rsvhn_cifarpol_wrn_2810 wrn_28_10 0.05 0.05 160 |& tee rsvhn_cifarpol_wrn2810.txt
./scripts/svhn.sh train-reduced svhn/svhn_2_23_b_policy_15 rsvhn_svhnpol_ss96 shake_shake_96 0.005 0.005 160 |& tee rsvhn_svhnpol_ss96.txt
./scripts/svhn.sh train-reduced svhn/svhn_2_23_b_policy_15 rsvhn_svhnpol_ss96 shake_shake_96 0.005 0.005 1760 |& tee rsvhn_svhnpol_ss96.txt
