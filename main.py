from data.dataReduction import DataReduction

data_reduction = DataReduction("D:/lipreading/graduation/dataset")
data = data_reduction.data_reduce()
print(data)
print(data.shape)