

#!/bin/bash

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90
resnet20 54
}

pruning_scratch_vgg(){
CUDA_VISIBLE_DEVICES=$1 python ./VGG/pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 
}
pruning_pretrain_160_precfg_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain --use_state_dict --lr 0.1 --epochs 160 \
--use_precfg
}

pruning_pretrain_160_mycfg_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain --use_state_dict --lr 0.1 --epochs 160
}
pruning_pretrain_40_precfg_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain --use_state_dict --lr 0.001 --epochs 40
--use_precfg
}

pruning_pretrain_40_mycfg_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 \
--use_pretrain --use_state_dict --lr 0.001 --epochs 40
}


pretrain_my_method_precfg(){
(pruning_pretrain_40_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch40_varience1 1 0.2)&
(pruning_pretrain_40_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch40_varience2 1 0.2)&
(pruning_pretrain_40_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch40_varience3 1 0.2)&


(pruning_pretrain_160_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch160_varience1 1 0.2)&
(pruning_pretrain_160_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch160_varience2 1 0.2)&
(pruning_pretrain_160_precfg_vgg 0 ./logs/vgg_pretrain/prune_precfg_epoch160_varience3 1 0.2)&
}
scratch_my_method_precfg(){
(pruning_scratch_vgg 0 ./logs/vgg_prune_precfg_varience4 1 0.2)&
(pruning_scratch_vgg 0 ./logs/vgg_prune_precfg_varience5 1 0.2)&
(pruning_scratch_vgg 0 ./logs/vgg_prune_precfg_varience6 1 0.2)&
}
scratch_my_method_mycfg(){

(pruning_scratch_vgg 0 ./logs/vgg_prune_ratenorm1_ratedist0.2_varience1 1 0.2)&
(pruning_scratch_vgg 0 ./logs/vgg_prune_ratenorm1_ratedist0.2_varience2 1 0.2)&
(pruning_scratch_vgg 0 ./logs/vgg_prune_ratenorm1_ratedist0.2_varience3 1 0.2)&

}



scratch_my_method_precfg
