#!/bin/bash

baseline(){
main_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  main_cifar_vgg_log.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2
}
#run baseline
(main_vgg 0 ./logs/baseline/main_1)&
(main_vgg 0 ./logs/baseline/main_2)&
(main_vgg 0 ./logs/baseline/main_3)&
}

prune_pretrain_and_finetune(){
#prune_pretrain
python PFEC_vggprune.py --model ./logs/PFEC/finetune_40_varience1/checkpoint.pth.tar --dataset cifar10 --save ./logs/PFEC/finetune_testacc

#finetune
(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune40_varience1)&
(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune40_varience2)&
(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune40_varience3)&

(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune160_varience1  --epochs 160)&
(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune160_varience2  --epochs 160)&
(python PFEC_finetune.py --refine ./logs/PFEC/prune/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --save ./logs/PFEC/finetune160_varience3  --epochs 160)&

eval_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  pruning_cifar_vgg.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--evaluate \
--save_path $2 \
--rate_norm $3 \
--rate_dist $4 \
--resume $5
}
#eval finetune model
eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience1/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience1/checkpoint.pth.tar
eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience2/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience2/checkpoint.pth.tar
eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience3/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune160_varience3/checkpoint.pth.tar

eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience1/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience1/checkpoint.pth.tar
eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience2/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience2/checkpoint.pth.tar
eval_vgg 0 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience3/ 1 0.2 /home/yahe/compress/filter_similarity/logs/PFEC/finetune40_varience3/checkpoint.pth.tar
}

prune_from_scratch(){
scratch_full_vgg(){
CUDA_VISIBLE_DEVICES=$1 python  main_cifar_vgg_log.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 --use_scratch
}
#get full scratch model
scratch_full_vgg 0 ./logs/off/scratch/full_model
#prune full scratch
python PFEC_vggprune.py --model ./logs/PFEC/scratch/full_model/checkpoint.pth.tar --dataset cifar10 --save ./logs/PFEC/scratch/pruned_model
}


run_finetune_scratch(){
#finetune small scratch model
finetune_scratch(){
CUDA_VISIBLE_DEVICES=$1 python  main_cifar_vgg_log.py  ./data/cifar.python --dataset cifar10 --arch vgg \
--save_path $2 \
--use_scratch --train_scratch $3
}
finetune_scratch 0  ./logs/PFEC/scratch/train_small_model ./logs/PFEC/scratch/pruned_model/pruned.pth.tar
}

#run baseline
baseline

#prune the pre-trained VGG and fine-tune for 40 or 160 epochs
prune_pretrain_and_finetune

#prune from scratch VGG
prune_from_scratch

#finetune from scratch VGG
run_finetune_scratch

