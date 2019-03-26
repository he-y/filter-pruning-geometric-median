import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch
import time

__all__ = ['ResNet_small', 'resnet18_small', 'resnet34_small', 'resnet50_small', 'resnet101_small', 'resnet152_small']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes_after_prune, planes_expand, planes_before_prune, index, bn_value, stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes_after_prune, stride)
        self.bn1 = nn.BatchNorm2d(planes_after_prune)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes_after_prune, planes_after_prune)
        self.bn2 = nn.BatchNorm2d(planes_after_prune)
        self.downsample = downsample
        self.stride = stride

        # for residual index match
        self.index = Variable(index)
        # for bn add
        self.bn_value = bn_value

        # self.out = torch.autograd.Variable(
        #     torch.rand(batch, self.planes_before_prune, 64 * 56 // self.planes_before_prune,
        #                64 * 56 // self.planes_before_prune), volatile=True).cuda()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: without index match
        # out += residual
        # out = self.relu(out)

        # setting: with index match
        residual += self.bn_value.cuda()
        residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


class Bottleneck(nn.Module):
    # expansion is not accurately equals to 4
    expansion = 4

    def __init__(self, inplanes, planes_after_prune, planes_expand, planes_before_prune, index, bn_value, stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes_after_prune, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes_after_prune)
        self.conv2 = nn.Conv2d(planes_after_prune, planes_after_prune, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes_after_prune)

        # setting: for accuracy expansion
        self.conv3 = nn.Conv2d(planes_after_prune, planes_expand, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes_expand)

        # setting: original resnet, expansion = 4
        # self.conv3 = nn.Conv2d(planes, planes_before_prune * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes_before_prune * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # for residual index match
        self.index = Variable(index)
        # for bn add
        self.bn_value = bn_value

        # self.extend = torch.autograd.Variable(
        #     torch.rand(self.planes_before_prune * 4, 64 * 56 // self.planes_before_prune,
        #                64 * 56 // self.planes_before_prune), volatile=True).cuda()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: without index match
        # print("residual size{},out size{} ".format(residual.size(),   out.size()))
        # out += residual
        # out = self.relu(out)

        # setting: with index match
        residual += self.bn_value.cuda()
        residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResNet_small(nn.Module):

    def __init__(self, block, layers, index, bn_value,
                 num_for_construct=[64, 64, 64 * 4, 128, 128 * 4, 256, 256 * 4, 512, 512 * 4],
                 num_classes=1000):
        super(ResNet_small, self).__init__()
        self.inplanes = num_for_construct[0]

        self.conv1 = nn.Conv2d(3, num_for_construct[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_for_construct[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # setting: expansion = 4
        # self.layer1 = self._make_layer(block, num_for_construct[1], num_for_construct[1] * 4, 64, index,  layers[0])
        # self.layer2 = self._make_layer(block, num_for_construct[2], num_for_construct[2] * 4, 128, index,  layers[1], stride=2)
        # self.layer3 = self._make_layer(block, num_for_construct[3], num_for_construct[3] * 4, 256, index,  layers[2], stride=2)
        # self.layer4 = self._make_layer(block, num_for_construct[4], num_for_construct[4] * 4, 512, index,  layers[3], stride=2)

        # setting: expansion may not accuracy equal to 4
        self.index_layer1 = {key: index[key] for key in index.keys() if 'layer1' in key}
        self.index_layer2 = {key: index[key] for key in index.keys() if 'layer2' in key}
        self.index_layer3 = {key: index[key] for key in index.keys() if 'layer3' in key}
        self.index_layer4 = {key: index[key] for key in index.keys() if 'layer4' in key}
        self.bn_layer1 = {key: bn_value[key] for key in bn_value.keys() if 'layer1' in key}
        self.bn_layer2 = {key: bn_value[key] for key in bn_value.keys() if 'layer2' in key}
        self.bn_layer3 = {key: bn_value[key] for key in bn_value.keys() if 'layer3' in key}
        self.bn_layer4 = {key: bn_value[key] for key in bn_value.keys() if 'layer4' in key}
        # print("bn_layer1", bn_layer1.keys(), bn_layer2.keys(), bn_layer3.keys(), bn_layer4.keys())

        self.layer1 = self._make_layer(block, num_for_construct[1], num_for_construct[2], 64, self.index_layer1, self.bn_layer1,
                                       layers[0])
        self.layer2 = self._make_layer(block, num_for_construct[3], num_for_construct[4], 128, self.index_layer2, self.bn_layer2,
                                       layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_for_construct[5], num_for_construct[6], 256, self.index_layer3, self.bn_layer3,
                                       layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_for_construct[7], num_for_construct[8], 512, self.index_layer4, self.bn_layer4,
                                       layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes_after_prune, planes_expand, planes_before_prune, index, bn_layer, blocks,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes_before_prune * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes_before_prune * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes_before_prune * block.expansion),
            )
        print("before pruning is {}, after pruning is {}:".format(planes_before_prune,planes_after_prune))

        # setting: accu number for_construct expansion
        index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv3' in key}
        index_block_0_value = list(index_block_0_dict.values())[0]

        bn_layer_0_value = list(bn_layer.values())[0]

        layers = []
        layers.append(
            block(self.inplanes, planes_after_prune, planes_expand, planes_before_prune, index_block_0_value,
                  bn_layer_0_value,
                  stride, downsample))
        #        self.inplanes = planes * block.expansion
        self.inplanes = planes_before_prune * block.expansion

        for i in range(1, blocks):
            index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv3') in key}
            index_block_i_value = list(index_block_i_dict.values())[0]

            bn_layer_i = {key: bn_layer[key] for key in bn_layer.keys() if (str(i) + '.bn3') in key}
            bn_layer_i_value = list(bn_layer_i.values())[0]
            layers.append(
                block(self.inplanes, planes_after_prune, planes_expand, planes_before_prune, index_block_i_value,
                      bn_layer_i_value,
                      ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
