#!/bin/bash

# @D&T:    Sun 06 Jan 2019 15:33:22 AEDT



#below is the core command

vgg_style1_pretrain(){
CUDA_VISIBLE_DEVICES=0,1,2,3 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a vgg16_bn --save_dir ./vgg_snap/vgg16bn_style1_pretrain \
 --VGG_pruned_style CP_5x --lr 0.01 --rate_norm 1 --rate_dist 0.3   \
 --use_pretrain \
 --workers 16 -b 256 
}


vgg_style1_scratch(){
CUDA_VISIBLE_DEVICES=0,1,2,3 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a vgg16_bn --save_dir ./vgg_snap/vgg16bn_style1_scratch \
 --VGG_pruned_style CP_5x --lr 0.01 --rate_norm 1 --rate_dist 0.3   \
 --workers 16 -b 256
}

vgg_style2_pretrain(){
CUDA_VISIBLE_DEVICES=0,1,2,3 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a vgg16_bn --save_dir ./vgg_snap/vgg16bn_style2_pretrain \
 --VGG_pruned_style Thinet_conv --lr 0.01 --rate_norm 1 --rate_dist 0.3   \
 --use_pretrain \
 --workers 16 -b 256
}

vgg_style2_scratch(){
CUDA_VISIBLE_DEVICES=0,1,2,3 python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012 -a vgg16_bn --save_dir ./vgg_snap/vgg16bn_style2_scratch \
 --VGG_pruned_style Thinet_conv --lr 0.01 --rate_norm 1 --rate_dist 0.3   \
 --workers 16 -b 256
}

vgg_baseline(){
CUDA_VISIBLE_DEVICES=0,1,2,3 python original_train.py /data/yahe/imagenet/ImageNet2012 -a vgg16_bn --save_dir ./vgg_snap/baseline_128  \
 --lr 0.01 --workers 48 -b 128
}




explain(){
change -workers == number of core of machine
change -b == 128  for_official_batch_size
change '/data/yahe/imagenet/ImageNet2012' to Imagenet directory
change 'CUDA_VISIBLE_DEVICES=0' to 'CUDA_VISIBLE_DEVICES=0,1,2,3' for_multiple_gpu 

move below four commander out of function to run it: 
vgg_style1_pretrain
vgg_style1_scratch
vgg_style2_pretrain
vgg_style2_scratch
}

vgg_baseline


