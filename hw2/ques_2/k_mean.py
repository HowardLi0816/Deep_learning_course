import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import math
import matplotlib.pyplot as plt
from dataset import get_data
import pandas as pd

def k_means(data, K):
    #whitened = whiten(data)
    whitened = data
    center, _ = kmeans(whitened, K)
    result, _ = vq(whitened, center)
    return center, result

def to_one_hot(result, K):
    size = result.shape[0]
    labels = np.zeros((size, K))
    for i in range(size):
        labels[i, result[i]] = 1
    return labels


if __name__ == '__main__':
    filename = './cluster.txt'
    data = get_data(filename)
    data_head = np.array(data[0])
    data_left = np.array(data[1])
    data_right = np.array(data[2])
    #print(data_head.shape, data_left.shape, data_right.shape)
    data_np = np.concatenate((data_head, data_left, data_right), axis=0)
    #print(data_np.shape)
    data_fea = data_np[:, :-1]
    K = 3
    acc = 0
    thresh = 0.5
    data_label = data_np[:, -1]
    while acc < thresh:
        center, result = k_means(data_fea, K)
        confusion_matrix = pd.crosstab(data_label, result, rownames=['Actual'], colnames=['Predicted'])
        conf = confusion_matrix.values
        acc = np.sum(np.diag(conf, k = 0)) / np.sum(conf)
        #print(a)


    #print(result)
    print(center)

    cl_0 = []
    cl_1 = []
    cl_2 = []
    for i in range(len(result)):
        if result[i] == 0:
            cl_0.append(data_fea[i])
        elif result[i] == 1:
            cl_1.append(data_fea[i])
        elif result[i] == 2:
            cl_2.append(data_fea[i])
    cl_data = [cl_0, cl_1, cl_2]

    fir = []
    sec = []
    for i in range(K):
        fir.append([])
        sec.append([])

    color = ['blue', 'red', 'green']
    for i in range(K):
        for j in range(len(cl_data[i])):
            fir[i].append(cl_data[i][j][0])
        for j in range(len(cl_data[i])):
            sec[i].append(cl_data[i][j][1])
        plt.scatter(fir[i], sec[i], c = color[i], label = i)
    plt.plot(center[:, 0], center[:, 1], 'k*', label='centroids')
    plt.legend()
    plt.show()
    counts = np.bincount(result)
    #print(counts)


    print(confusion_matrix)
    print("acc:", acc)

    one_hot = to_one_hot(result, K)
    #print(one_hot)



