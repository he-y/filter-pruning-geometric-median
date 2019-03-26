# Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration

**[CVPR 2019 Oral](https://arxiv.org/abs/1811.00250)**. Implementation with PyTorch.

This implementation is based on [soft-filter-pruning](https://github.com/he-y/soft-filter-pruning).

## Table of Contents

- [Requirements](#requirements)
- [Models and log files](#models-and-log-files)
- [Training ResNet on ImageNet](#training-imagenet)
  - [Usage of Pruning Training](#usage-of-pruning-training)
  - [Usage of Normal Training](#usage-of-normal-training)
  - [Inference the pruned model with zeros](#inference-the-pruned-model-with-zeros)
  - [Inference the pruned model without zeros](#inference-the-pruned-model-without-zeros)
  - [Scripts to reproduce the results in our paper](#scripts-to-reproduce-the-results-in-our-paper)
- [Training ResNet on Cifar-10](#training-resnet-on-cifar-10)
- [Training VGGNet on Cifar-10](#training-vggnet-on-cifar-10)
- [Notes](#notes)
  - [Torchvision Version](#torchvision-version)
  - [Why use 100 epochs for training](#why-use-100-epochs-for-training)
  - [Process of ImageNet dataset](#process-of-imagenet-dataset)
  - [FLOPs Calculation](#flops-calculation)
- [Citation](#citation)


## Requirements
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.3.0

## Models and log files
The trained models with log files can be found in [Google Drive](https://drive.google.com/drive/folders/1lPhInbd7v3HjK9uOPW_VNjGWWm7kyS8e?usp=sharing).

The pruned model without zeros: [Release page](https://github.com/he-y/soft-filter-pruning/releases/tag/ResNet50_pruned).

## Training ResNet on ImageNet

#### Usage of Pruning Training
We train each model from scratch by default. If you wish to train the model with pre-trained models, please use the options `--use_pretrain --lr 0.01`. 

Run Pruning Training ResNet (depth 152,101,50,34,18) on Imagenet:

```bash
python pruning_train.py -a resnet152 --save_path ./snapshots/resnet152-rate-0.7 --rate_norm 1 --rate_dist 0.4 --layer_begin 0 --layer_end 462 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet101 --save_path ./snapshots/resnet101-rate-0.7 --rate_norm 1 --rate_dist 0.4 --layer_begin 0 --layer_end 309 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet50  --save_path ./snapshots/resnet50-rate-0.7 --rate_norm 1 --rate_dist 0.4 --layer_begin 0 --layer_end 156 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet34  --save_path ./snapshots/resnet34-rate-0.7 --rate_norm 1 --rate_dist 0.4 --layer_begin 0 --layer_end 105 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet18  --save_path ./snapshots/resnet18-rate-0.7 --rate_norm 1 --rate_dist 0.4 --layer_begin 0 --layer_end 57 --layer_inter 3  /path/to/Imagenet2012
```
Explanation:
 
Note1: `rate_norm = 0.9` means pruning 10% filters by norm-based criterion, `rate_dist = 0.2` means pruning 20% filters by distance-based criterion.

Note2: the `layer_begin` and `layer_end` is the index of the first and last conv layer, `layer_inter` choose the conv layer instead of BN layer. 

#### Usage of Normal Training
Run resnet(100 epochs): 
```bash
python original_train.py -a resnet50 --save_dir ./snapshots/resnet50-baseline  /path/to/Imagenet2012 --workers 36
```

#### Inference the pruned model with zeros
```bash
sh function/inference_pruned.sh
```

#### Scripts to reproduce the results in our paper
To train the ImageNet model with / without pruning, see the directory `scripts`.
Full script is [here](https://github.com/he-y/filter_similarity/blob/master/scripts/pruning_imagenet_resnet.sh).


## Training ResNet on Cifar-10
```bash
sh scripts/pruning_cifar10.sh
```
Please be care of the hyper-parameter [`layer_end`](https://github.com/he-y/filter_similarity/blob/master/scripts/pruning_cifar10.sh#L4-L9) for different layer of ResNet.

Reproduce ablation study of Cifar-10:
```bash
sh scripts/ablation_pruning_cifar10.sh
```


## Training VGGNet on Cifar-10
Refer to the directory `VGG_cifar`. 
#### Reproduce previous paper [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
```bash
sh VGG_cifar/scripts/PFEC_train_prune.sh
```
Four function included in the script, including [training baseline](https://github.com/he-y/filter_similarity/tree/master/VGG_cifar/scripts#L3-L12), [pruning from pretrain](https://github.com/he-y/filter_similarity/tree/master/VGG_cifar/scripts#L14-L43), [pruning from scratch](https://github.com/he-y/filter_similarity/tree/master/VGG_cifar/scripts#L45-L54), [finetune the pruend](https://github.com/he-y/filter_similarity/tree/master/VGG_cifar/scripts#L57-L65)

#### Our method
```bash
sh VGG_cifar/scripts/pruning_vgg_my_method.sh
```
Including [pruning the pretrained](https://github.com/he-y/filter_similarity/blob/master/VGG_cifar/scripts/pruning_vgg_my_method.sh#L52-L61), [pruning the scratch](https://github.com/he-y/filter_similarity/blob/master/VGG_cifar/scripts/pruning_vgg_my_method.sh#L62-L66).

## Notes

#### Torchvision Version
We use the torchvision of 0.3.0. If the version of your torchvision is 0.2.0, then the `transforms.RandomResizedCrop` should be `transforms.RandomSizedCrop` and the `transforms.Resize` should be `transforms.Scale`.

#### Why use 100 epochs for training
This can improve the accuracy slightly.

#### Process of ImageNet dataset
We follow the [Facebook process of ImageNet](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
Two subfolders ("train" and "val") are included in the "/path/to/ImageNet2012".
The correspding code is [here](https://github.com/he-y/filter_similarity/blob/master/pruning_imagenet.py#L136-L137).

#### FLOPs Calculation
Refer to the [file](https://github.com/he-y/soft-filter-pruning/blob/master/utils/cifar_resnet_flop.py).



## Citation
```
@inproceedings{he2019filter,
  title     = {Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration},
  author    = {He, Yang and Liu, Ping and Wang, Ziwei and Hu, Zhilan and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```
