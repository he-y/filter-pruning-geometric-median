"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnext29_16_64 = models.ResNeXt29_16_64(num_classes)
    resnext29_8_64 = models.ResNeXt29_8_64(num_classes)
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .resnext import resnext29_8_64, resnext29_16_64
from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet_mod import resnet_mod20, resnet_mod32, resnet_mod44, resnet_mod56, resnet_mod110

from .preresnet import preresnet20, preresnet32, preresnet44, preresnet56, preresnet110
from .caffe_cifar import caffe_cifar
from .densenet import densenet100_12

# imagenet based resnet
from .imagenet_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# cifar based resnet
from .resnet import CifarResNet, ResNetBasicblock

# cifar based resnet pruned
from .resnet_small import resnet20_small, resnet32_small, resnet44_small, resnet56_small, resnet110_small
# imagenet based resnet pruned
# from .imagenet_resnet_small import resnet18_small, resnet34_small, resnet50_small, resnet101_small, resnet152_small
from .imagenet_resnet_small import resnet18_small, resnet34_small, resnet50_small, resnet101_small, resnet152_small


from .vgg_cifar10 import *
from .vgg import *
