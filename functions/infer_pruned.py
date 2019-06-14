# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os
import shutil
import pdb, time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import convert_secs2time, time_string, time_file_str
import models
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--eval_small', dest='eval_small', action='store_true', help='whether a big or small model')
parser.add_argument('--small_model', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()


def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'gpu-time.{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=False)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
            state_dict = remove_module_dict(state_dict)
            model.load_state_dict(state_dict)
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Scale(256),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.evaluate:
        print_log("eval true", log)
        if not args.eval_small:
            big_model = model.cuda()
            print_log('Evaluate: big model', log)
            print_log('big model accu: {}'.format(validate(val_loader, big_model, criterion, log)), log)
        else:
            print_log('Evaluate: small model', log)
            if args.small_model:
                if os.path.isfile(args.small_model):
                    print_log("=> loading small model '{}'".format(args.small_model), log)
                    small_model = torch.load(args.small_model)
                    for x, y in zip(small_model.named_parameters(), model.named_parameters()):
                        print_log("name of layer: {}\n\t *** small model {}\n\t *** big model {}".format(x[0], x[1].size(),
                                                                                                         y[1].size()), log)
                    if args.use_cuda:
                        small_model = small_model.cuda()
                    print_log('small model accu: {}'.format(validate(val_loader, small_model, criterion, log)), log)
                else:
                    print_log("=> no small model found at '{}'".format(args.small_model), log)
        return


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        if args.use_cuda:
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def remove_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = name[7:] if name.startswith("module.") else name  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    main()
