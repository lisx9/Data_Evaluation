import numpy as np
from data.LSH import EuclideanLSH
import torch
import statistics
data = np.random.random([10000, 5])
query = np.random.random([5])
lsh = EuclideanLSH(8, 8, 5)
lsh.insert(data)
a = lsh.hash_tables[0]
print(a)
a = np.array(a)
print(torch.Tensor(a))
