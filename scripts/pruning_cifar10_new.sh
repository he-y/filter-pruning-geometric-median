#!/bin/bash
echo "Pruning with filter similarity!"

pruning_scratch(){

declare -A NUM_LAYER_END
NUM_LAYER_END["resnet110"]="324"
NUM_LAYER_END["resnet56"]="162"
NUM_LAYER_END["resnet32"]="90"
NUM_LAYER_END["resnet20"]="52"

GPU_IDS=$1
SAVE_PATH=$2
RATE_NORM=$3
RATE_DIST=$4
MODEL=$5

CUDA_VISIBLE_DEVICES=$GPU_IDS python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch $MODEL \
--save_path $SAVE_PATH \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $RATE_NORM \
--rate_dist $RATE_DIST \
--layer_begin 0  --layer_end ${NUM_LAYER_END[$MODEL]} --layer_inter 3 --epoch_prune 1
}


pruning_pretrain(){

GPU_IDS=$1
SAVE_PATH=$2
RATE_NORM=$3
RATE_DIST=$4
MODEL=$5

CUDA_VISIBLE_DEVICES=$GPU_IDS python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch $MODEL \
--save_path $SAVE_PATH \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $RATE_NORM \
--rate_dist $RATE_DIST \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end ${NUM_LAYER_END[$MODEL]} --layer_inter 3 --epoch_prune 1
}



declare -a MODELS=('resnet110' 'resnet56'  'resnet32' 'resnet20')
NUM_RUNS=1
GPU_ID=0
RATE_NORM=1
RATE_DIST=0.4

mkdir ./checkpoints/

for model in "${MODELS[@]}"
do
  for idx in `seq 1 $NUM_RUNS`;
  do
    pruning_scratch  $GPU_ID ./checkpoints/$model'_ratenorm'$RATE_NORM'_ratedist'$RATE_DIST'_varience'$idx  1 0.4  $model
  done
done











pruning_pretrain(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.001 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet56(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet56 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 164 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet56(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet56 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 164 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet32(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet32 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 90 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet32(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet32 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 90 --layer_inter 3 --epoch_prune 1
}


pruning_scratch_resnet20(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet20 \
--save_path $2 \
--epochs 200 \
--schedule  60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1
}
pruning_pretrain_resnet20(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch resnet20 \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 \
--gammas 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain \
--use_state_dict \
--layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1
}





