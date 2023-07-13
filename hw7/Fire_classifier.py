import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from time import time
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path
from torchvision.models import resnet34

DATASETS = {
    'fire': {
        'num_classes': 3,
        'img_size': 224,
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

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['fire'],
                        default='fire')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    #parser.add_argument('--decay_lr', default=False, type=str2bool)
    #parser.add_argument('--decay_position',  nargs='+', type=int, default=[25])
    parser.add_argument('--shuffle', default=True, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['Adam'], default='Adam')
    parser.add_argument('--reg', type=float, default=1e-4) #default L2 regularization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--model', type=str, default='ResNet34', choices=['ResNet34'])
    parser.add_argument('--unfreeze_epoch', type=int, default=5)
    parser.add_argument('--lr_decay', type=int, default=5)
    parser.add_argument('--plot_learning_curve', type=str2bool, default=True)
    parser.add_argument('--confusion_matrix', type=str2bool, default=True)

    return parser


def accuracy(output, target, args):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_epoch(device, trainloader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0
    if epoch < args.unfreeze_epoch:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    for batch_idx, (images, target) in enumerate(trainloader):

        images, target = images.to(device), target.to(device)

        output = model(images)
        #print (f"output: {output.shape}")

        loss = criterion(output, target)

        acc1 = accuracy(output, target, args)
        num_input_sum += images.shape[0]
        loss_sum += float(loss.item() * images.shape[0])
        acc1_num_sum += float(acc1[0] * images.shape[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #scheduler.step()


        if batch_idx % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
    return avg_loss, avg_acc1

def validate(device, testloader, model, criterion, epoch, args, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    pred = np.array([])
    tar = np.array([])
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(testloader):

            images, target = images.to(device), target.to(device)

            output = model(images)
            loss = criterion(output, target)

            # for confusion matrix
            pred = np.concatenate((pred, np.argmax(output.cpu().detach().numpy(), axis=1)))
            if args.dataset == 'mnist':
                tar = np.concatenate((tar, np.argmax(target.cpu().detach().numpy(), axis=1)))
            else:
                tar = np.concatenate((tar, target.cpu().detach().numpy()))

            acc1 = accuracy(output, target, args)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_loss, avg_acc, pred, tar

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch == args.lr_decay:
        lr *= 0.1 * lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_checkpoint(checkpoint_pthpath):
    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]


def train(args):
    #use gpus
    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    print (f'config: {args}')

    if args.dataset == 'fire':
        img_size = 224

    if args.dataset in DATASETS:
    #    img_size=DATASETS[args.dataset]['img_size']
        num_classes = DATASETS[args.dataset]['num_classes']
        img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']

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

    if args.dataset == 'fire':
        DATA_PTH = './data/S1_Raw_Photographs_Full_Study/'
        data_set = torchvision.datasets.ImageFolder(root=DATA_PTH, transform=augmentations)
        train_set, val_set, test_set = torch.utils.data.random_split(data_set, [int(0.7 * 3000), int(0.15 * 3000), int(0.15 * 3000)])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    #define Optimizer & Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)



    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)

    #start training
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    time_begin = time()
    best_val_acc = 0
    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train_epoch(device, trainloader, model, criterion, optimizer, epoch, args)
        val_loss, val_acc, _, _ = validate(device, valloader, model, criterion, epoch, args, time_begin)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)


        if val_acc > best_val_acc:
            print(f"save model")
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"./model/best_model_dataset_{args.dataset}.pth",
            )

    print(f'config: {args}')
    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_val_acc:.2f}, '
          f'final top-1: {val_acc:.2f}')

    if args.plot_learning_curve:
        plt.figure()
        x = np.arange(1, args.max_epoch+1)
        plt.plot(x, train_acc_list, label="train_acc", color='b')
        plt.ylabel("acc")
        plt.title("Train and validation acc vs iterations")
        plt.plot(x, val_acc_list, label="validation acc", color='r')
        plt.plot(args.lr_decay, 0, 'o', label="lr_decay and unfreeze point", color='g')
        plt.legend()
        plt.show()

        plt.figure()
        x = np.arange(1, args.max_epoch + 1)
        plt.plot(x, train_loss_list, label="train_log_loss", color='b')
        plt.ylabel("log_loss")
        plt.title("Train and validation log_loss vs iterations")
        plt.plot(x, val_loss_list, label="validation_log_loss", color='r')
        plt.plot(args.lr_decay, 0, 'o', label="lr_decay and unfreeze point", color='g')
        plt.legend()
        plt.show()

    model_pred = resnet34(pretrained=False)
    model_pred.fc = nn.Linear(model_pred.fc.in_features, num_classes)

    model_pred.to(device)
    model_state_dict, optimizer_state_dict = load_checkpoint(f'./model/best_model_dataset_{args.dataset}.pth')
    model_pred.load_state_dict(model_state_dict)

    if args.confusion_matrix:
        _, test_acc, pred, tar = validate(device, testloader, model_pred, criterion, epoch, args, time_begin)
        cm = confusion_matrix(tar, pred)
        conf_matrix = pd.DataFrame(cm, index=['0', '1', '2'],
                                   columns=['0', '1', '2'])
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues")
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

        print("Test Acc = ", test_acc)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP training script')

    parser=init_parser(parser)

    args = parser.parse_args()

    train(args)

