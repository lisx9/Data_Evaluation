import os
import torch
from data.dataLoader import MyDataset
from data.LSH import EuclideanLSH
import numpy as np
import math
import random

class DataReduction:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def data_reduce(self):
        file_size = len(os.listdir(self.data_folder + "/video"))
        X = []
        y = []
        tables_num = 8   #哈希个数
        a = 1        #桶容量
        depth = 1     #向量维度
        my_datasset = MyDataset(self.data_folder, transform=False, convert_gray=False)
        for i in range(file_size):
            video, label = my_datasset.__getitem__(i)
            video = video.transpose(1, 3)   #(T, C, H, W)
            video = video[:8]
            feature = my_datasset.get_swin_feature(video)
            depth = feature.shape[0]
            X.append(feature)
            y.append(label)
        X = torch.stack(X, dim=0)
        X = X.detach().cpu().numpy()
        lsh = EuclideanLSH(tables_num, a, depth)
        lsh.insert(X)
        data = []
        tables = lsh.hash_tables
        n_samples = self.get_samples(tables, file_size)
        for i in range(len(tables)):
            res = random.sample(tables[i], n_samples[i])
            for j in res:
                data.append(j)
        data = np.array(data)
        return torch.Tensor(data)
    def get_samples(self, data, N):
        '''
        :param data: lsh分桶后的数据
        :param N: 数据总量
        :return: 抽取的样本
        '''
        n = [len(x) for x in data]
        s = [self.get_deviation(x) for x in data]
        n_s = [a * b for a, b in zip(n, s)]
        n_samples = [int(N * a / sum(n_s)) if int(N * a / sum(n_s)) > 0 else 1 for a in n_s]
        return n_samples


    def get_deviation(self, data):
        vec_sum = np.zeros(data[0].shape)
        for i in data:
            vec_sum = np.add(vec_sum, i)
        bar_x = vec_sum / len(data)
        cov_sum = 0
        for i in data:
            res = [(a - b)**2 for a, b in zip(i, bar_x)]
            cov_sum += sum(res)
        cov_sum = cov_sum / (len(data) - 1)
        return math.sqrt(cov_sum)