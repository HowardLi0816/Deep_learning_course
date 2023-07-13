from PIL import ImageDraw, ImageFont, Image
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
#from vit_model import *
import torchvision
from torchvision.transforms import ToTensor, transforms
import torch.utils.data as data
from torch.autograd import Variable
from sklearn import metrics
from hparams import *
import collections
from sklearn.metrics import roc_auc_score
import random
import logging
import sys
from timm.models import create_model, safe_model_name, resume_checkpoint, \
    convert_splitbn_model, model_parameters
from torchvision.models import resnet34
from model.CNN5 import CNN5

from timm.loss import LabelSmoothingCrossEntropy
import os
from time import time
import argparse
import math
from utils.helpers import load_checkpoint


DATASETS = {
    'asl': {
        'num_classes': 29,
        'img_size': 200,
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2719, 0.2654, 0.2743]
    }
}
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_parser(parser):

    parser.add_argument('--device', type=str, default='gpu')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['asl'],
                        default='asl')

    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)

    parser.add_argument('--model', type=str, default='CNN5', choices=['ResNet34', 'CNN5'])

    parser.add_argument('--train_batch_used', default=1000000, type=int)
    parser.add_argument('--val_batch_used', default=1000000, type=int)
    parser.add_argument('--load_pretrain_model', type=str2bool, default=True) # True

    parser.add_argument('--lr', default=6e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--regularization', default=6e-2, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--batch_size', default=256, type=int) # 512
    parser.add_argument('--print_freq', default=200, type=int)

    return parser


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

#def load_pretrain(model, path):


#def train(device, trainloader, model, criterion, optimizer, scheduler, epoch, args):
def train_with_KD(device, trainloader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0

    if args.model == 'ResNet34':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True


    for batch_idx, (images, target) in enumerate(trainloader):
        if batch_idx>args.train_batch_used:
            break

        images, target = images.to(device), target.to(device)

        output = model(images)
        # print (f"output: {output.shape}")

        loss = criterion(output, target)

        acc1 = accuracy(output, target)
        num_input_sum += images.shape[0]
        loss_sum += float(loss.item() * images.shape[0])
        acc1_num_sum += float(acc1[0] * images.shape[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # scheduler.step()


        if batch_idx % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def validate(device, testloader, model, criterion, epoch, args, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(testloader):
            if batch_idx > args.val_batch_used:
                break

            images, target = images.to(device), target.to(device)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    #elif not args.disable_cos:
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_top_module(args):
    set_seed(42)

    global a_logger

    a_logger = logging.getLogger()
    a_logger.setLevel(logging.DEBUG)

    output_file_handler = logging.FileHandler(
        f"output_{args.model}_{args.dataset}.log")

    stdout_handler = logging.StreamHandler(sys.stdout)
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)

    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    print (f'config: {args}')


    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']


    img_size = args.img_size

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = []
    augmentations += [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size // 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        *normalize,
    ])
    augmentations = transforms.Compose(augmentations)


    if args.dataset == 'asl':
        num_classes = 29
        DATA_PTH = './data/archive/asl_alphabet_train/asl_alphabet_train/'
        data_set = torchvision.datasets.ImageFolder(root=DATA_PTH, transform=augmentations)
        trainset, valset = torch.utils.data.random_split(data_set, [int(0.85 * 87000), int(0.15 * 87000)])

    #print (f"Size of trainset: {len(trainset)}")
    #print (f"Size of testset: {len(testset)}")


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    testloader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=2)

    print (f"Size of trainloader: {len(trainloader)}")
    print (f"Size of testloader: {len(testloader)}")

    if args.model=="ResNet34":
        model = resnet34(pretrained=args.load_pretrain_model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model=='CNN5':
        model = CNN5()
    #print (model)
    #print (model.classifier.blocks[0].self_attn.qkv.weight.grad)

    model.to(device)


    epochs = args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.regularization)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.warmup_epoch)

    criterion = LabelSmoothingCrossEntropy()
    criterion.to(device)

    if not os.path.exists('./cnn5_model'):
        os.makedirs('./cnn5_model')

    time_begin = time()
    best_val_acc = 0
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, args)
        #train(device, trainloader, model, criterion, optimizer, scheduler, epoch, args)
        train_with_KD(device, trainloader, model, criterion, optimizer, epoch, args)
        val_acc=validate(device, testloader, model, criterion, epoch, args, time_begin)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"./cnn5_model/best_{args.model}_pretrain_{args.load_pretrain_model}_imgsize_{img_size}_bs_{args.batch_size}.pth",
            )

    print (f'config: {args}')
    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_val_acc:.2f}, '
          f'final top-1: {val_acc:.2f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASL quick training script')




    parser=init_parser(parser)

    args = parser.parse_args()

    #hparams = PARAMS
    #hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)


    train_top_module(args)




