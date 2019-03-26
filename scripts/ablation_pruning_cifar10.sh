#!/bin/bash
echo "Pruning with filter similarity!"

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90
resnet20 54
}


pruning_scratch_resnet110_abalation_epoch(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune $5
}


pruning_scratch_resnet110_abalation_rate(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}

pruning_scratch_resnet110_abalation_dist(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1 \
--dist_type $5
}

pruning_pretrain_resnet110(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}

run_dist(){


(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl2_varience1 1 0.4 l2)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl2_varience2 1 0.4 l2)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl2_varience3 1 0.4 l2)&


(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl1_varience1 1 0.4 l1)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl1_varience2 1 0.4 l1)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distl1_varience3 1 0.4 l1)&


(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distcos_varience1 1 0.4 cos)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distcos_varience2 1 0.4 cos)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_distcos_varience3 1 0.4 cos)&


wait
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl2_varience1 1 0.3 l2)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl2_varience2 1 0.3 l2)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl2_varience3 1 0.3 l2)&


(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl1_varience1 1 0.3 l1)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl1_varience2 1 0.3 l1)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distl1_varience3 1 0.3 l1)&


(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distcos_varience1 1 0.3 cos)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distcos_varience2 1 0.3 cos)&
(pruning_scratch_resnet110_abalation_dist 0 /data/yahe/cifar_GM/abalition/dist_fast/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_distcos_varience3 1 0.3 cos)&
}

run_epoch(){

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience1 1 0.4 1)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience2 1 0.4 1)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience3 1 0.4 1)&


(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch2_varience1 1 0.4 2)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch2_varience2 1 0.4 2)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch2_varience3 1 0.4 2)&


(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch3_varience1 1 0.4 3)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch3_varience2 1 0.4 3)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch3_varience3 1 0.4 3)&

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch4_varience1 1 0.4 4)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch4_varience2 1 0.4 4)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch4_varience3 1 0.4 4)&
wait
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch5_varience1 1 0.4 5)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch5_varience2 1 0.4 5)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch5_varience3 1 0.4 5)&

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch6_varience1 1 0.4 6)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch6_varience2 1 0.4 6)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch6_varience3 1 0.4 6)&

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch7_varience1 1 0.4 7)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch7_varience2 1 0.4 7)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch7_varience3 1 0.4 7)&
wait
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch8_varience1 1 0.4 8)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch8_varience2 1 0.4 8)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch8_varience3 1 0.4 8)&

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch9_varience1 1 0.4 9)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch9_varience2 1 0.4 9)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch9_varience3 1 0.4 9)&

(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch10_varience1 1 0.4 10)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch10_varience2 1 0.4 10)&
(pruning_scratch_resnet110_abalation_epoch 0 /data/yahe/cifar_GM/abalition/epoch/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch10_varience3 1 0.4 10)&

}
run_rate(){

(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.1_epoch1_varience1 1 0.1)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.1_epoch1_varience2 1 0.1)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.1_epoch1_varience3 1 0.1)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.2_epoch1_varience1 1 0.2)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.2_epoch1_varience2 1 0.2)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.2_epoch1_varience3 1 0.2)&
wait
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_epoch1_varience1 1 0.3)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_epoch1_varience2 1 0.3)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.3_epoch1_varience3 1 0.3)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience1 1 0.4)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience2 1 0.4)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.4_epoch1_varience3 1 0.4)&
wait
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.5_epoch1_varience1 1 0.5)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.5_epoch1_varience2 1 0.5)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.5_epoch1_varience3 1 0.5)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.6_epoch1_varience1 1 0.6)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.6_epoch1_varience2 1 0.6)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.6_epoch1_varience3 1 0.6)&
wait
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.7_epoch1_varience1 1 0.7)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.7_epoch1_varience2 1 0.7)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.7_epoch1_varience3 1 0.7)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.8_epoch1_varience1 1 0.8)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.8_epoch1_varience2 1 0.8)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.8_epoch1_varience3 1 0.8)&
wait
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.9_epoch1_varience1 1 0.9)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.9_epoch1_varience2 1 0.9)&
(pruning_scratch_resnet110_abalation_rate 0 /data/yahe/cifar_GM/abalition/rate/scartch_cifar10_resnet110_ratenorm1_ratedist0.9_epoch1_varience3 1 0.9)&

}
