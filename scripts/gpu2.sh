#!/usr/bin/env bash


python pruning_imagenet.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar  --save_dir ./gpu/resnet50-0.7-time/ --batch-size 64 \
-e --eval_small


#python gpu_time.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  \
#--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar  --save_dir ./gpu/resnet50-0.7-time/ --batch-size 64 --big_small


