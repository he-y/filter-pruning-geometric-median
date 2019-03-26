from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

## http://torch.ch/blog/2015/07/30/cifar.html
class CifarCaffeNet(nn.Module):
  def __init__(self, num_classes):
    super(CifarCaffeNet, self).__init__()

    self.num_classes = num_classes

    self.block_1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.ReLU(),
      nn.BatchNorm2d(32))

    self.block_2 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(64))

    self.block_3 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(128))

    self.classifier = nn.Linear(128*9, self.num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def forward(self, x):
    x = self.block_1.forward(x)
    x = self.block_2.forward(x)
    x = self.block_3.forward(x)
    x = x.view(x.size(0), -1)
    #print ('{}'.format(x.size()))
    return self.classifier(x)

def caffe_cifar(num_classes=10):
  model = CifarCaffeNet(num_classes)
  return model
