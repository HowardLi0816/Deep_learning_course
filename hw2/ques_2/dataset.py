import numpy as np
import re

def get_data(filename):
    data_head = []
    data_left = []
    data_right = []
    data = [data_head, data_left, data_right]

    file = open(filename, 'r')
    file_data = file.readlines()
    r_head = r"\d+\.\d+\ \d+\.\d+\ Head"
    r_left = r"\d+\.\d+\ \d+\.\d+\ Ear_left"
    r_right = r"\d+\.\d+\ \d+\.\d+\ Ear_right"
    for row in file_data:
        match_head = re.search(r_head, row)
        match_left = re.search(r_left, row)
        match_right = re.search(r_right, row)
        if match_head != None:
            tmp_list = row.split(' ')
            tmp_list[0] = float(tmp_list[0])
            tmp_list[1] = float(tmp_list[1])
            #tmp_list[2] = tmp_list[2].rstrip('\n')
            tmp_list[2] = int(0)
            data_head.append(tmp_list)
        if match_left != None:
            tmp_list = row.split(' ')
            tmp_list[0] = float(tmp_list[0])
            tmp_list[1] = float(tmp_list[1])
            #tmp_list[2] = tmp_list[2].rstrip('\n')
            tmp_list[2] = int(1)
            data_left.append(tmp_list)
        if match_right != None:
            tmp_list = row.split(' ')
            tmp_list[0] = float(tmp_list[0])
            tmp_list[1] = float(tmp_list[1])
            #tmp_list[2] = tmp_list[2].rstrip('\n')
            tmp_list[2] = int(2)
            data_right.append(tmp_list)
    return data

if __name__ == "__main__":
    filename = './cluster.txt'
    data = get_data(filename)
    print(data)
    data_head = np.array(data[0])
    data_left = np.array(data[1])
    data_right = np.array(data[2])
    print(data_head.shape, data_left.shape, data_right.shape)
    data_np = np.concatenate((data_head, data_left, data_right), axis = 0)
    print(data_np.shape)

