import numpy as np
from typing import List, Union
class EuclideanLSH:

    def __init__(self, tables_num: int, a: int, depth: int):
        """

        :param tables_num: hash_table的个数
        :param a: a越大，被纳入同个位置的向量就越多，即可以提高原来相似的向量映射到同个位置的概率，
                反之，则可以降低原来不相似的向量映射到同个位置的概率。
        :param depth: 向量的维度数
        """
        self.tables_num = tables_num
        self.a = a
        # 为了方便矩阵运算，调整了shape，每一列代表一个hash_table的随机向量
        self.R = np.random.random([depth, tables_num])
        self.b = np.random.uniform(0, a, [1, tables_num])
        # 初始化空的hash_table
        self.hash_tables = []

    def _hash(self, inputs: Union[List[List], np.ndarray]):
        """
        将向量映射到对应的hash_table的索引
        :param inputs: 输入的单个或多个向量
        :return: 每一行代表一个向量输出的所有索引，每一列代表位于一个hash_table中的索引
        """
        # H(V) = |V·R + b| / a，R是一个随机向量，a是桶宽，b是一个在[0,a]之间均匀分布的随机变量
        hash_val = np.floor(np.abs(np.matmul(inputs, self.R) + self.b) / self.a)

        return hash_val

    def insert(self, inputs):
        """
        将向量映射到对应的hash_table的索引，并插入到所有hash_table中
        :param inputs:
        :return:
        """
        # 将inputs转化为二维向量
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        hash_index = self._hash(inputs)
        hash_code = []
        for i in hash_index:
            i = map(str, i)
            code = ''.join(i)
            hash_code.append(code)
        index = list(set(hash_code))
        for i in range(len(index)):
            l = []
            for j in range(len(hash_code)):
                if index[i] == hash_code[j]:
                    l.append(inputs[j])
            self.hash_tables.append(l)