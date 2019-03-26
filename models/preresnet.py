import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .res_utils import DownsampleA, DownsampleC
import math


class ResNetBasicblock(nn.Module):
  expansion = 1
  def __init__(self, inplanes, planes, stride, downsample, Type):
    super(ResNetBasicblock, self).__init__()

    self.Type = Type

    self.bn_a = nn.BatchNorm2d(inplanes)
    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

    self.bn_b = nn.BatchNorm2d(planes)
    self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.bn_a(x)
    basicblock = self.relu(basicblock)

    if self.Type == 'both_preact':
      residual = basicblock
    elif self.Type != 'normal':
      assert False, 'Unknow type : {}'.format(self.Type)

    basicblock = self.conv_a(basicblock)

    basicblock = self.bn_b(basicblock)
    basicblock = self.relu(basicblock)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(residual)
    
    return residual + basicblock

class CifarPreResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarPreResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarPreResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes

    self.conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
    self.lastact = nn.Sequential(nn.BatchNorm2d(64*block.expansion), nn.ReLU(inplace=True))
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(64*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, 'both_preact'))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1, None, 'normal'))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_3x3(x)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.lastact(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

def preresnet20(num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarPreResNet(ResNetBasicblock, 20, num_classes)
  return model

def preresnet32(num_classes=10):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarPreResNet(ResNetBasicblock, 32, num_classes)
  return model

def preresnet44(num_classes=10):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarPreResNet(ResNetBasicblock, 44, num_classes)
  return model

def preresnet56(num_classes=10):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarPreResNet(ResNetBasicblock, 56, num_classes)
  return model

def preresnet110(num_classes=10):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarPreResNet(ResNetBasicblock, 110, num_classes)
  return model
