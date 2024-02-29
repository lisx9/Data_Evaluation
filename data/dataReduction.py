import os
import torch
from data.dataLoader import MyDataset
from data.LSH import EuclideanLSH
import statistics
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
        bucket_size = []
        standard_deviation = []
        n_sample = []
        N = 0
        for i in range(tables_num):
            n_i = len(lsh.hash_tables[i])
            s_i = statistics.stdev(lsh.hash_tables[i].values())
            bucket_size.append(n_i)
            standard_deviation.append(s_i)
            N += n_i * s_i
        for i in range(tables_num):
            num = int(file_size * bucket_size[i] * standard_deviation[i] / N)
            n_sample.append(num)
        data = []
        for i in range(tables_num):
            sampled_subset = random.choices(lsh.hash_tables[i], k=n_sample[i])
            data.append(sampled_subset)
        return torch.Tensor(data)