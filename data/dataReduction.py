from dataLoader import MyDataset

my_datasset = MyDataset("D:/lipreading/graduation/dataset", transform=False)
video, label = my_datasset.__getitem__(0)
print(video, label)