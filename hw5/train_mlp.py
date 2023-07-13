import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import random
import math

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
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--decay_lr', default=False, type=str2bool)
    parser.add_argument('--decay_position',  nargs='+', type=int, default=[25])
    parser.add_argument('--shuffle', default=True, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['SGD'], default='SGD')
    parser.add_argument('--plot_learning_curve', default=True, type=str2bool)
    parser.add_argument('--reg', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--layer_param', nargs='+', type=int, default=[784, 200, 10])
    parser.add_argument('--validation_rate', type=int, default=6)
    parser.add_argument('--is_test', type=str2bool, default=False)

    return parser

def gen_data(args):
    if args.dataset == 'mnist':
        train_dir = './data/mnist_traindata.hdf5'
        test_dir = './data/mnist_testdata.hdf5'
        with h5py.File(train_dir, 'r+') as f:
            train_fea = f['xdata'][:]
            train_labels = f['ydata'][:]

        with h5py.File(test_dir, 'r+') as ft:
            test_fea = ft['xdata'][:]
            test_labels = ft['ydata'][:]

        train_labels = train_labels.argmax(axis=1)
        test_labels = test_labels.argmax(axis=1)

        train = np.concatenate((train_fea, train_labels.reshape(train_labels.shape[0], 1)), axis=1)
        test = np.concatenate((test_fea, test_labels.reshape(test_labels.shape[0], 1)), axis=1)

    return train, test

def split_train(train, args):
    start_idx = random.randint(0, args.validation_rate)
    valid = train[start_idx:train.shape[0]:args.validation_rate, :]
    after_va_train = np.delete(train, np.arange(start_idx, train.shape[0], args.validation_rate), axis=0)
    return after_va_train, valid

def softmax(x):
    max_item = np.max(x, axis=0)
    norm_matrix = np.tile(max_item.reshape(max_item.shape[0], 1), x.shape[0]).T
    return np.exp(x - norm_matrix) / np.sum(np.exp(x - norm_matrix), axis=0)


def relu(x):
    return np.where(x < 0, 0, x)

def drelu(x):
    return np.where(x < 0, 0, 1)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanh(x):
    return 1 - x ** 2

class MLP(object):
    def __init__(self, args):
        self.lr = args.lr
        self.max_epoch = args.max_epoch
        self.num_class = args.class_num
        self.batch_size = args.batch_size
        self.layer = [np.zeros((args.layer_param[0], self.batch_size))]
        self.weights = []
        self.bias = []

        #initial weight
        for i in range(len(args.layer_param)-1):
            self.weights.append(np.random.randn(args.layer_param[i+1], args.layer_param[i]))
            self.bias.append(np.random.randn(args.layer_param[i+1]))
            self.layer.append(np.ones((args.layer_param[i+1], self.batch_size)))

        #hidden layer activation function
        if args.activation == 'relu':
            self.activation = relu
            self.deactivation = drelu
        elif args.activation == 'tanh':
            self.activation = tanh
            self.deactivation = dtanh

            #classification layer activation function
        self.classifier_act = softmax

    def forward(self, data_fea, args):
        data_num = len(data_fea)
        self.layer[0] = data_fea.T
        for i in range(len(args.layer_param)-1):
            if i != len(args.layer_param)-2:
                self.layer[i+1] = self.activation(np.dot(self.weights[i], self.layer[i]) + np.tile(self.bias[i].reshape(self.bias[i].shape[0], 1), data_num))
            elif i == len(args.layer_param)-2:
                self.layer[i + 1] = self.classifier_act(np.dot(self.weights[i], self.layer[i]) + np.tile(self.bias[i].reshape(self.bias[i].shape[0], 1), data_num))

    def validate(self, data_fea, labels):
        self.forward(data_fea, args)
        pred_label = np.argmax(self.layer[-1], axis=0)
        acc = np.sum((pred_label == labels)) / len(data_fea)
        return pred_label, acc

    def train(self, data_fea, labels, val_fea, val_labels, args):

        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        best_acc = 0
        best_epoch = 0
        count = 0

        for epoch in range(self.max_epoch):
            if args.decay_lr:
                if epoch in args.decay_position:
                    self.lr /= 2
            if args.shuffle == True:
                idx = np.random.permutation(len(data_fea))
                data_fea = data_fea[idx]
                labels = labels[idx]
            batches = math.ceil(len(data_fea) / self.batch_size)
            loss_sum = 0
            acc_sum = 0
            for batch in range(batches):
                if batch == batches-1:
                    batch_fea = data_fea[batch * self.batch_size:, :]
                    batch_label = labels[batch * self.batch_size:]
                else:
                    batch_fea = data_fea[batch * self.batch_size: (batch + 1) * self.batch_size, :]
                    batch_label = labels[batch * self.batch_size: (batch + 1) * self.batch_size]

                one_hot = np.zeros((args.class_num, len(batch_label)))
                one_hot[batch_label.astype('int8'), np.arange(len(batch_label)).astype('int8')] = 1

                self.forward(batch_fea, args)
                self.backprop(one_hot, args)

                _, acc = self.validate(batch_fea, batch_label)
                self.criterion(batch_fea, batch_label, args)
                loss_sum += self.loss
                acc_sum += acc

                if (batch + 1) % args.print_freq == 0:
                    batch_loss, batch_acc = (loss_sum / args.print_freq), (acc_sum / args.print_freq)
                    print(f'[Epoch {epoch + 1}][Train][{batch + 1}] \t Loss: {batch_loss:.4e} \t Acc {batch_acc:6.4f} \t lr {self.lr:.4e}')
                    train_loss_list.append(batch_loss)
                    train_acc_list.append(batch_acc)
                    _, val_ba_acc = self.validate(val_fea, val_labels)
                    self.criterion(val_fea, val_labels, args)
                    val_ba_loss = self.loss
                    val_loss_list.append(val_ba_loss)
                    val_acc_list.append(val_ba_acc)
                    count += 1
                    loss_sum = 0
                    acc_sum = 0

            _, val_acc = self.validate(val_fea, val_labels)
            self.criterion(val_fea, val_labels, args)
            val_loss = self.loss
            print(f'[Epoch {epoch + 1}][Eval] \t Loss: {val_loss:.4e} \t Acc {val_acc:6.4f}')

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_weight = self.weights
                best_bias = self.bias

        self.weights = best_weight
        self.bias = best_bias
        print(f'Script finished, best epoch: {best_epoch} '
            f'best validation acc: {best_acc:.4f}, ')

        if args.plot_learning_curve:
            plt.figure()
            x = np.arange(1, count + 1)
            plt.plot(x, train_acc_list, label="train_acc", color='b')
            plt.ylabel("acc")
            plt.title("Train and validation/test acc vs iterations")
            plt.plot(x, val_acc_list, label="validation/test_acc", color='r')
            plt.plot(args.decay_position, [0, 0], 'bo', label="decay position")
            plt.legend()
            plt.show()


    def backprop(self, one_hot_labels, args):
        #one_hot_labels:(class_num, batch_size)
        delta = self.layer[-1] - one_hot_labels
        for lay in range(len(args.layer_param)-1, 0, -1):
            grad_b = np.sum(delta, axis=1) / self.batch_size + 2 * args.reg * self.bias[lay-1]
            grad_w = np.dot(delta, self.layer[lay-1].T) / self.batch_size + 2 * args.reg * self.weights[lay-1]
            delta = self.deactivation(self.layer[lay-1]) * np.dot(self.weights[lay-1].T, delta)
            self.weights[lay-1] -= self.lr * grad_w
            self.bias[lay-1] -= self.lr * grad_b

    def criterion(self, data_fea, labels, args):
        pred_label, _ = self.validate(data_fea, labels)
        norm = 0
        for i in range(len(self.weights)):
            norm += np.sqrt(np.linalg.norm(self.weights[i]))
            norm += np.sqrt(np.linalg.norm(self.bias[i]))
        self.loss = -1 * np.sum(np.log(self.layer[-1][labels.astype('int8'), np.arange(len(data_fea))]+1e-5)) / len(data_fea) + args.reg * norm

def train_top(args):
    model = MLP(args)
    train, test = gen_data(args)
    after_va_train, valid = split_train(train, args)
    if not args.is_test:
        model.train(after_va_train[:, :-1], after_va_train[:, -1], valid[:, :-1], valid[:, -1], args)
    else:
        model.train(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1], args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP training')

    parser = init_parser(parser)

    args = parser.parse_args()

    train_top(args)