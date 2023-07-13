import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import splev, splrep
import math
import random
import matplotlib.pyplot as plt


def load_raman_data(filename):
    raw_data = pd.read_table(filename)
    raw_data = raw_data.to_numpy()
    #print("data shape", raw_data.shape)
    return raw_data

def find_n_peak(data, dis, hei):
    '''
    len, wid = data.shape
    dis = random.randint(1, len-1)
    peak_num = math.inf
    i = 0
    while peak_num != n_peak:
        peaks, _ = find_peaks(data[:, -1], distance=dis)
        peak_num = peaks.shape[0]
        print(dis, peak_num)
        if peak_num < n_peak:
            dis = int(dis / 2)
        elif peak_num > n_peak:
            dis = int(dis * 3 / 2)
        i += 1
        #print(i)
    best_dis = dis
    '''
    peaks, _ = find_peaks(data[:, -1], distance=dis, height=hei)
    peaks_point = data[peaks]
    peaks_point = np.concatenate((peaks_point, peaks.reshape(peaks.shape[0], 1)), axis=1)
    return peaks_point


def plot_interpolation(data):
    spl = splrep(data[:, 0], data[:, -1])
    #print(spl)
    x2 = np.linspace(data[0, 0], data[-1, 0], 1000)
    y2 = splev(x2, spl)
    plt.plot(data[:, 0], data[:, -1], 'o', x2, y2)

    intensity_der = splev(x2, spl, der=1)
    #print(intensity_der)
    plt.plot(x2, intensity_der)

    idx = 0
    for i in range(intensity_der.shape[0]-1):
        if intensity_der[i] * intensity_der[i+1] < 0:
            idx = i

    zero_point = x2[idx] + (x2[idx+1]-x2[idx]) * abs(intensity_der[idx]) / (abs(intensity_der[idx]) + abs(intensity_der[idx+1]))
    #print(zero_point)
    y = splev(zero_point, spl)
    plt.plot(zero_point, y, 'D')
    plt.legend(['Original', 'Spline', 'Spline Derivative', 'zero crossing'])
    plt.show()


if __name__ == "__main__":
    data_file = './raman.txt'
    raw_data = load_raman_data(data_file)
    peak_8 = find_n_peak(raw_data, 150, 1500)
    peak_sort_idx = np.argsort(peak_8[:, 1])
    peak_sort_idx = peak_sort_idx[::-1]
    peak_sort = peak_8[peak_sort_idx]
    #print(peak_8)
    print(peak_sort)
    plt.plot(raw_data[:, 0], raw_data[:, 1])
    plt.plot(peak_8[:, 0], peak_8[:, 1], 'x')
    plt.show()

    n = 1
    for i in range(4):
        if i == 0:
            n = 10
        elif i == 1:
            n = 10
        elif i == 2:
            n = 1
        elif i == 3:
            n = 1
        data_point = peak_sort[i, :-1]
        tmp = True
        idx = np.zeros(2)
        for j in range(raw_data.shape[0]):
            if raw_data[j, 0] < data_point[0]-n:
                idx[0] = j+1
            if raw_data[j, 0] > data_point[0]+n and tmp:
                idx[1] = j
                tmp = False
        #print(idx)
        ran = raw_data[int(idx[0]):int(idx[1]), :]
        plot_interpolation(ran)



