#!/bin/bash
echo "Pruning with filter similarity!"

change_layer_end_for_different_structure(){
resnet152 462
resnet101 309
resnet50 156
resnet34 105
resnet18 57
}


pruning_pretrain_res18(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet18 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 57 --layer_inter 3 --workers 12 \
--use_pretrain
}

pruning_scratch_res18(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet18 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 57 --layer_inter 3 --workers 12
}


pruning_pretrain_res34(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet34 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 105 --layer_inter 3 --workers 36 \
--use_pretrain
}

pruning_scratch_res34(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet34 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 105 --layer_inter 3 --workers 36
}


pruning_pretrain_res50(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet50 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 156 --layer_inter 3 --workers 36 \
--use_pretrain
}

pruning_scratch_res50(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet50 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 156 --layer_inter 3 --workers 36
}

pruning_pretrain_res101(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet101 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 309 --layer_inter 3 --workers 12 \
--use_pretrain
}

pruning_scratch_res101(){
CUDA_VISIBLE_DEVICES=$1 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a resnet101 \
--save_dir $2 \
--rate_norm $3 \
--rate_dist $4 \
--layer_begin 0 --layer_end 309 --layer_inter 3 --workers 12
}


run_res18(){
(pruning_pretrain_res18 0 ./log/imagenet/pretrain-resnet18-ratenorm1-ratedist0.4 1 0.4)&
(pruning_scratch_res18 1 ./log/imagenet/scratch-resnet18-ratenorm1-ratedist0.4 1 0.4)&

(pruning_pretrain_res18 0 ./log/imagenet/pretrain-resnet18-ratenorm1-ratedist0.3 1 0.3)&
(pruning_scratch_res18 1 ./log/imagenet/scratch-resnet18-ratenorm1-ratedist0.3 1 0.3)&
}

run_res34(){
(pruning_pretrain_res34 6 ./log/imagenet/pretrain-resnet34-ratenorm1-ratedist0.4 1 0.4)&
(pruning_scratch_res34  7 ./log/imagenet/scratch-resnet34-ratenorm1-ratedist0.4 1 0.4)&

(pruning_pretrain_res34 6 ./log/imagenet/pretrain-resnet34-ratenorm1-ratedist0.3 1 0.3)&
(pruning_scratch_res34  7 ./log/imagenet/scratch-resnet34-ratenorm1-ratedist0.3 1 0.3)&

(pruning_pretrain_res34  3 ./log/imagenet/pretrain-resnet34-ratenorm0.7-ratedist0.1 0.7 0.1)&
(pruning_scratch_res34  3 ./log/imagenet/scratch-resnet34-ratenorm0.7-ratedist0.1 0.7 0.1)&

(pruning_pretrain_res34  3 ./log/imagenet/pretrain-resnet34-ratenorm0.8-ratedist0.1 0.8 0.1)&
(pruning_scratch_res34  3 ./log/imagenet/scratch-resnet34-ratenorm0.8-ratedist0.1 0.8 0.1)&
}

run_res50(){
pruning_pretrain_res50 0,1,2 ./log/imagenet/pretrain-resnet50-ratenorm1-ratedist0.4 1 0.4
pruning_scratch_res50  0,1,2 ./log/imagenet/scratch-resnet50-ratenorm1-ratedist0.4 1 0.4

pruning_pretrain_res50 0,1,2 ./log/imagenet/pretrain-resnet50-ratenorm1-ratedist0.3 1 0.3
pruning_scratch_res50  0,1,2 ./log/imagenet/scratch-resnet50-ratenorm1-ratedist0.3 1 0.3


pruning_pretrain_res50 0,1,2 ./log/imagenet/pretrain-resnet50-ratenorm0.8-ratedist0.1 0.8 0.1
pruning_scratch_res50  0,1,2 ./log/imagenet/scratch-resnet50-ratenorm0.8-ratedist0.1 0.8 0.1

pruning_pretrain_res50 0,1,2 ./log/imagenet/pretrain-resnet50-ratenorm0.7-ratedist0.1 0.7 0.1
pruning_scratch_res50  0,1,2 ./log/imagenet/scratch-resnet50-ratenorm0.7-ratedist0.1 0.7 0.1

pruning_pretrain_res50 0,1,2 ./log/imagenet/pretrain-resnet50-ratenorm0.6-ratedist0.1 0.6 0.1
pruning_scratch_res50  0,1,2 ./log/imagenet/scratch-resnet50-ratenorm0.6-ratedist0.1 0.6 0.1

}

run_res101(){

pruning_pretrain_res101 3,4,5,6,7,0,1,2 ./log/imagenet/pretrain-resnet101-ratenorm1-ratedist0.4 1 0.4
pruning_scratch_res101 3,4,5,6,7,0,1,2 ./log/imagenet/scratch-resnet101-ratenorm1-ratedist0.4 1 0.4

pruning_scratch_res101 3,4,5,6,7,0,1,2 ./log/imagenet/scratch-resnet101-ratenorm1-ratedist0.3 1 0.3

pruning_pretrain_res101 3,4,5,6,7,0,1,2 ./log/imagenet/pretrain-resnet101-ratenorm0.8-ratedist0.1 0.8 0.1
pruning_scratch_res101  3,4,5,6,7,0,1,2 ./log/imagenet/scratch-resnet101-ratenorm0.8-ratedist0.1 0.8 0.1

pruning_pretrain_res101 3,4,5,6,7,0,1,2 ./log/imagenet/pretrain-resnet101-ratenorm0.7-ratedist0.1 0.7 0.1
pruning_scratch_res101  3,4,5,6,7,0,1,2 ./log/imagenet/scratch-resnet101-ratenorm0.7-ratedist0.1 0.7 0.1

}
