import math

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['vgg']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg(depth=16)
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)
    a = []
    for x, y in enumerate(net.named_parameters()):
        print(x, y[0], y[1].size())
    #
    # for index, m in enumerate(net.modules()):
    #     print(index,m)
    #     if isinstance(m, nn.Conv2d):
    #         print("conv",index, m)
    # import numpy as np
    # cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
    #
    # cfg_mask = []
    # layer_id = 0
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         out_channels = m.weight.data.shape[0]
    #         if out_channels == cfg[layer_id]:
    #             cfg_mask.append(torch.ones(out_channels))
    #             layer_id += 1
    #             continue
    #         weight_copy = m.weight.data.abs().clone()
    #         weight_copy = weight_copy.cpu().numpy()
    #         L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
    #         arg_max = np.argsort(L1_norm)
    #         arg_max_rev = arg_max[::-1][:cfg[layer_id]]
    #         assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
    #         mask = torch.zeros(out_channels)
    #         mask[arg_max_rev.tolist()] = 1
    #         cfg_mask.append(mask)
    #         layer_id += 1
    #     elif isinstance(m, nn.MaxPool2d):
    #         layer_id += 1
    #
    # newmodel = vgg(dataset='cifar10', cfg=cfg)
    # newmodel.cuda()
    #
    # start_mask = torch.ones(3)
    # layer_id_in_cfg = 0
    # end_mask = cfg_mask[layer_id_in_cfg]
    # for [m0, m1] in zip(net.modules(), newmodel.modules()):
    #     if isinstance(m0, nn.BatchNorm2d):
    #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
    #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
    #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
    #         m1.running_var = m0.running_var[idx1.tolist()].clone()
    #         layer_id_in_cfg += 1
    #         start_mask = end_mask
    #         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
    #             end_mask = cfg_mask[layer_id_in_cfg]
    #     elif isinstance(m0, nn.Conv2d):
    #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
    #         w1 = w1[idx1.tolist(), :, :, :].clone()
    #         m1.weight.data = w1.clone()
    #     elif isinstance(m0, nn.Linear):
    #         if layer_id_in_cfg == len(cfg_mask):
    #             idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
    #             if idx0.size == 1:
    #                 idx0 = np.resize(idx0, (1,))
    #             m1.weight.data = m0.weight.data[:, idx0].clone()
    #             m1.bias.data = m0.bias.data.clone()
    #             layer_id_in_cfg += 1
    #             continue
    #         m1.weight.data = m0.weight.data.clone()
    #         m1.bias.data = m0.bias.data.clone()
    #     elif isinstance(m0, nn.BatchNorm1d):
    #         m1.weight.data = m0.weight.data.clone()
    #         m1.bias.data = m0.bias.data.clone()
    #         m1.running_mean = m0.running_mean.clone()
    #         m1.running_var = m0.running_var.clone()
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         a.append(m)
    #         print(m)
    print(1)
