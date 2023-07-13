import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import h5py

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
                        choices=['mnist'],
                        default='mnist')
    parser.add_argument('--max_epoch', default=1000, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--min_lr', default=1e-3, type=float)
    parser.add_argument('--warmup', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--adjust_lr', default=False, type=str2bool)
    parser.add_argument('--shuffle', default=False, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['BGD', 'SGD'], default='BGD')
    parser.add_argument('--plot_learning_curve', default=False, type=str2bool)
    parser.add_argument('--binary_class', type=str2bool, default=True)
    parser.add_argument('--binary_num', type=int, default=2)
    parser.add_argument('--reg_type', type=str, choices=['l1', 'l2'], default='l2')
    parser.add_argument('--reg_coeff', type=float, default=1e-2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--class_num', type=int, default=10)

    return parser

def gen_data(args):
    if args.dataset == 'mnist':
        train_dir = '../data/mnist_traindata.hdf5'
        test_dir = '../data/mnist_testdata.hdf5'
        with h5py.File(train_dir, 'r+') as f:
            train_fea = f['xdata'][:]
            train_labels = f['ydata'][:]

        with h5py.File(test_dir, 'r+') as ft:
            test_fea = ft['xdata'][:]
            test_labels = ft['ydata'][:]

        if args.binary_class:
            detect_class = args.binary_num
            train_labels = (train_labels.argmax(axis=1) == detect_class)
            test_labels = (test_labels.argmax(axis=1) == detect_class)
            #print(train_labels)
        else:
            train_labels = train_labels.argmax(axis=1)
            test_labels = test_labels.argmax(axis=1)

    return train_fea, train_labels, test_fea, test_labels

def sigmoid(x):
    '''
    if x>=0:
        return 1/(1+np.exp(x * -1))
    else:
        return np.exp(x)/(1+np.exp(x))
    '''

    return 1/(1+np.exp(x * -1))

def softmax(x):
    max_item = np.max(x, axis=0)
    norm_matrix = np.tile(max_item.reshape(max_item.shape[0], 1), x.shape[0]).T
    return np.exp(x - norm_matrix) / np.sum(np.exp(x - norm_matrix), axis=0)

def validate(xdata, target, weights, bias, args):
    # probabality and loss
    if args.binary_class:
        p_x = sigmoid(np.dot(xdata, weights) + bias)
        loss = -(np.sum(target * np.log(p_x + 10e-10) + (1 - target) * np.log(1 - p_x + 10e-10))) / len(xdata)

        # cal acc
        acc = np.sum((p_x >= args.threshold) == target) / len(xdata)
    else:
        #wx+b
        forward_cal = np.dot(weights, xdata.T) + np.tile(bias.reshape(bias.shape[0], 1), xdata.shape[0])
        #cal p_x, p_x is [class_num, batch_size]
        p_x = softmax(forward_cal)
        #cal loss
        loss = -1 * np.sum(np.log(p_x[target, np.arange(0, len(xdata))] + 10e-10)) / len(xdata)

        #cal acc
        pre_results = p_x.argmax(axis=0)
        acc = np.sum(pre_results == target) / len(xdata)

    return p_x, loss, acc

def regressor(weights, bias, xdata, target, lr, args):
    p_x, loss, acc = validate(xdata, target, weights, bias, args)

    #cal regularization
    reg_coeff = args.reg_coeff
    if args.reg_type == 'l1':
        grad_reg_w = reg_coeff * np.sign(weights)
        grad_reg_b = reg_coeff * np.sign(bias)
    elif args.reg_type == 'l2':
        grad_reg_w = 2 * reg_coeff * weights
        grad_reg_b = 2 * reg_coeff * bias

    #cal weight grad
    if args.binary_class:
        grad_z = p_x - target
        grad_weights = np.dot(xdata.T, grad_z) / len(xdata) + grad_reg_w
        grad_b = np.sum(grad_z) / len(xdata) + grad_reg_b
    else:
        p_x[target, np.arange(0, len(xdata))] = p_x[target, np.arange(0, len(xdata))] - 1
        grad_weights = np.dot(p_x, xdata) / len(xdata) + grad_reg_w
        grad_b = np.sum(p_x, axis=1) / len(xdata) + grad_reg_b

    #update new weights
    weights = weights - lr * grad_weights
    bias = bias - lr * grad_b

    return acc, loss, weights, bias


def init_param(fea_num, args):
    np.random.seed(0)
    if args.binary_class:
        init_w = np.random.rand(fea_num)
        init_b = np.random.random()
    else:
        init_w = np.random.rand(args.class_num, fea_num)
        init_b = np.random.rand(args.class_num)
    return init_w, init_b

def adjust(epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.max_epoch - args.warmup)))
    # elif not args.disable_cos:
    '''
    elif epoch > 100:
        lr = 0.95 * last_lr
    '''

    if lr < args.min_lr:
        lr = args.min_lr
    return lr

def train(args):
    train_fea, train_labels, test_fea, test_labels = gen_data(args)
    fea_num = train_fea.shape[1]
    weights, bias = init_param(fea_num, args)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_acc = 0
    best_epoch = 0
    count = 0

    for epoch in range(args.max_epoch):
        if args.adjust_lr:
            lr = adjust(epoch, args)
        else:
            lr = args.lr
        if args.train_method == 'BGD':
            count = args.max_epoch
            batch_size = train_fea.shape[0]
            _, test_loss, test_acc = validate(test_fea, test_labels, weights, bias, args)
            acc, loss, weights, bias = regressor(weights, bias, train_fea, train_labels, lr, args)
            print(f'[Epoch {epoch + 1}][Train] \t Loss: {loss:.4e} \t Acc {acc:6.4f} \t lr {lr:.4e}')
            train_loss_list.append(loss)
            train_acc_list.append(acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
        elif args.train_method == 'SGD':
            #shuffle
            idx = np.random.permutation(len(train_fea))
            train_fea = train_fea[idx]
            train_labels = train_labels[idx]
            batch_size = args.batch_size
            batch_num = math.ceil(len(train_fea) / batch_size)
            # for print loggging for training
            loss_sum = 0
            acc_sum = 0

            # validation for new weights for each epochs
            _, test_loss, test_acc = validate(test_fea, test_labels, weights, bias, args)
            for batch in range(batch_num):
                if batch != batch_num - 1:
                    batch_fea = train_fea[(batch * batch_size):(batch + 1) * batch_size]
                    batch_labels = train_labels[(batch * batch_size):(batch + 1) * batch_size]
                else:
                    batch_fea = train_fea[(batch * batch_size):]
                    batch_labels = train_labels[(batch * batch_size):]

                acc, loss, weights, bias = regressor(weights, bias, batch_fea, batch_labels, lr, args)
                loss_sum += loss
                acc_sum += acc

                # print logging during train
                if (batch + 1) % args.print_freq == 0:
                    batch_loss, batch_acc = (loss_sum / args.print_freq), (acc_sum / args.print_freq)
                    print(f'[Epoch {epoch + 1}][Train][{batch + 1}] \t Loss: {batch_loss:.4e} \t Acc {batch_acc:6.4f} \t lr {lr:.4e}')
                    train_loss_list.append(batch_loss)
                    train_acc_list.append(batch_acc)
                    _, te_ba_loss, te_ba_acc = validate(test_fea, test_labels, weights, bias, args)
                    test_loss_list.append(te_ba_loss)
                    test_acc_list.append(te_ba_acc)
                    count += 1
                    loss_sum = 0
                    acc_sum = 0

        print(f'[Epoch {epoch + 1}][Eval] \t Loss: {test_loss:.4e} \t Acc {test_acc:6.4f}')




        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            best_test_loss = test_loss
            best_train_acc = acc
            best_train_loss = loss
            outfile = f'./best_model/best_model_chenghao_li_dataset_{args.dataset}_binary_{args.binary_class}_{args.binary_num}_train_method_{args.train_method}_reg_type_{args.reg_type}_reg_coeff_{args.reg_coeff}_epoch_{args.max_epoch}_adjust_lr_{args.adjust_lr}_lr_{args.lr}_min_lr_{args.min_lr}_batch_{batch_size}.hd5'
            with h5py.File(outfile, 'w') as hf:
                hf.create_dataset('W', data=np.asarray(weights))
                hf.create_dataset('b', data=np.asarray(bias))
    print(f'Script finished, best epoch: {best_epoch} '
          f'best train loss: {best_train_loss:.4e}, '
          f'best train acc: {best_train_acc:.4f}, '
          f'best test loss: {best_test_loss:.4e}, '
          f'best test acc: {best_acc:.4f}, ')

    if args.plot_learning_curve:
        plt.figure()
        x = np.arange(1, count + 1)
        plt.plot(x, train_loss_list, label="train_loss", color='b')
        plt.ylabel("Loss")
        plt.title("Train and test loss vs iterations")
        plt.plot(x, test_loss_list, label="test_loss", color='r')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(x, train_acc_list, label="train_acc", color='b')
        plt.ylabel("acc")
        plt.title("Train and test acc vs iterations")
        plt.plot(x, test_acc_list, label="test_acc", color='r')
        plt.legend()
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mnist training')

    parser = init_parser(parser)

    args = parser.parse_args()

    train(args)