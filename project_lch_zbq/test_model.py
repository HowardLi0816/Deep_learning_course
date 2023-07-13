from PIL import ImageDraw, ImageFont, Image
import requests
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import torch
import torch.nn as nn
import numpy as np
from datasets import load_metric
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

from timm.loss import LabelSmoothingCrossEntropy
import os
from time import time
import argparse
import math
from utils.helpers import load_checkpoint
from model.CNN5 import CNN5
from model.CNN5_modify import CNN4


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

    parser.add_argument('--model', type=str, default='CNN5', choices=['ResNet34', 'CNN5', 'CNN4'])

    # parser.add_argument('--train_batch_used', default=1000000, type=int)
    # parser.add_argument('--val_batch_used', default=1000000, type=int)
    # parser.add_argument('--load_pretrain_model', type=str2bool, default=True) # True
    #
    # parser.add_argument('--lr', default=6e-4, type=float)
    # parser.add_argument('--min_lr', default=1e-5, type=float)
    # parser.add_argument('--warmup', default=1, type=int, help='number of warmup epochs')
    # parser.add_argument('--regularization', default=6e-2, type=float)
    # parser.add_argument('--epochs', default=600, type=int)
    # parser.add_argument('--batch_size', default=256, type=int) # 512
    parser.add_argument('--print_freq', default=200, type=int)
    parser.add_argument('--use_KD', default=False, type=str2bool)

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


def validate(device, testloader, model, criterion, args, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(testloader):

            images, target = images.to(device), target.to(device)
            if args.model == 'ResNet34' or args.model == 'CNN4':
                output = model(images)
            elif args.model == 'CNN5':
                output, _ = model(images)


            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Test][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin)
    print(f'[Test] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc

def test_model(args):
    set_seed(42)

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

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        *normalize,
    ])


    if args.dataset == 'asl':
        num_classes = 29
        # DATA_PTH = './data/archive/asl_alphabet_test_remove_back/asl_alphabet_test/'
        DATA_PTH = './data/archive/asl_alphabet_test_prem/asl_alphabet_test/'
        test_set = torchvision.datasets.ImageFolder(root=DATA_PTH, transform=transform_test)

    #print (f"Size of trainset: {len(trainset)}")
    #print (f"Size of testset: {len(testset)}")


    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=2)

    print (f"Size of testloader: {len(testloader)}")

    if args.model=="ResNet34":
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_path = f"./resnet_model/best_ResNet34_pretrain_True_imgsize_224_bs_32.pth"
        model_state_dict, optimizer_state_dict = load_checkpoint(model_path)
        model.load_state_dict(model_state_dict)

    elif args.model=="CNN5":
        model = CNN5()
        if args.use_KD:
            model_path = f"./best_model/best_KD_CNN5_ResNet34_imgsize_224_bs_32_epochs_10.pth"
            # model_path = f"./KD_model/best_KD_CNN5_ResNet34_imgsize_224_bs_16_epochs_10.pth"
        else:
            model_path = f"./best_model/best_CNN5_pretrain_True_imgsize_224_bs_32.pth"
        model_state_dict, optimizer_state_dict = load_checkpoint(model_path)
        model.load_state_dict(model_state_dict)

    elif args.model=="CNN4":
        model = CNN4(flatten=7)
        model_path = f"./best_model/best_CNN4_pretrain_True_imgsize_112_bs_32.pth"
        model_state_dict, optimizer_state_dict = load_checkpoint(model_path)
        model.load_state_dict(model_state_dict)
    #print (model)
    #print (model.classifier.blocks[0].self_attn.qkv.weight.grad)

    model.to(device)

    criterion = LabelSmoothingCrossEntropy()
    criterion.to(device)

    time_begin = time()

    test_acc=validate(device, testloader, model, criterion, args, time_begin)

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'test top-1: {test_acc:.2f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASL quick testing script')




    parser=init_parser(parser)

    args = parser.parse_args()

    #hparams = PARAMS
    #hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)


    test_model(args)




