#!/usr/bin/env bash


python infer_pruned.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar  --batch-size 64 \
-e --eval_small --save_dir ./logs/infer_small_model/ 

python infer_pruned.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar  --batch-size 64 \
-e --eval_small  --save_dir ./logs/infer_big_model/ 


