import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import argparse

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
    parser.add_argument('--plot_random_data', default=True, type=str2bool)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--plot_num', type=int, default=3)

    return parser

def read_weights(W_FNAME):
    with h5py.File(W_FNAME, 'r+') as hf:
        W_1 = hf['W1'][:]
        b_1 = hf['b1'][:]
        W_2 = hf['W2'][:]
        b_2 = hf['b2'][:]
        W_3 = hf['W3'][:]
        b_3 = hf['b3'][:]

    print('W_1 shape:', W_1.shape)
    print('b_1 shape:', b_1.shape)
    print('W_2 shape:', W_2.shape)
    print('b_2 shape:', b_2.shape)
    print('W_3 shape:', W_3.shape)
    print('b_3 shape:', b_3.shape)
    return W_1, b_1, W_2, b_2, W_3, b_3

def gen_data(args):
    if args.dataset == 'mnist':
        test_dir = './mnist_testdata.hdf5'
        with h5py.File(test_dir, 'r+') as ft:
            test_fea = ft['xdata'][:]
            test_labels = ft['ydata'][:]

            test_labels = test_labels.argmax(axis=1)

    return test_fea, test_labels

def softmax(x):
    max_item = np.max(x, axis=0)
    norm_matrix = np.tile(max_item.reshape(max_item.shape[0], 1), x.shape[0]).T
    return np.exp(x - norm_matrix) / np.sum(np.exp(x - norm_matrix), axis=0)

def relu(x):
    return x * (x>0)

def forwards(args):
    W_file = './mnist_network_params.hdf5'
    W_1, b_1, W_2, b_2, W_3, b_3 = read_weights(W_file)
    test_fea, test_labels = gen_data(args)
    data_num = len(test_fea)
    x = test_fea.T;

    #define model
    x = np.dot(W_1, x) + np.tile(b_1.reshape(b_1.shape[0], 1), data_num)
    x = relu(x)
    x = np.dot(W_2, x) + np.tile(b_2.reshape(b_2.shape[0], 1), data_num)
    x = relu(x)
    x = np.dot(W_3, x) + np.tile(b_3.reshape(b_3.shape[0], 1), data_num)
    x = softmax(x)

    pred = np.argmax(x, axis=0)

    json_data = []
    for i in range(data_num):
        json_data.append({'index': i, 'activations': x[:, i].tolist(), 'classification': pred[i].tolist()})

    #write json
    with open("./result.json", "w") as f:
        f.write(json.dumps(json_data))

    #compare with grounftruth
    compare = (pred == test_labels)
    true_pred = np.sum(compare)
    print('Success prediction number:', true_pred)

    if args.plot_random_data:
        ran_true_idx = np.array(np.where(compare==1)).reshape(true_pred)
        np.random.shuffle(ran_true_idx)
        ran_false_idx = np.array(np.where(compare==0)).reshape(data_num-true_pred)
        np.random.shuffle(ran_false_idx)
        for i in range(args.plot_num):
            plt.imshow(test_fea[ran_true_idx[i], :].reshape(28, 28))
            plt.title(f'No.{ran_true_idx[i]} image, True predicted, class:{test_labels[ran_true_idx[i]]}')
            plt.show()

            plt.imshow(test_fea[ran_false_idx[i], :].reshape(28, 28))
            plt.title(f'No.{ran_false_idx[i]} image, False predicted, class:{test_labels[ran_false_idx[i]]}, pred_class:{pred[ran_false_idx[i]]}')
            plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mnist feed forward')

    parser = init_parser(parser)

    args = parser.parse_args()

    forwards(args)