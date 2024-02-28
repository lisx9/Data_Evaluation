from dataLoader import MyDataset

my_datasset = MyDataset("D:/lipreading/graduation/dataset", transform=False)
video, label = my_datasset.__getitem__(1)
print(video, label)
print(video.shape)
feature = my_datasset.get_swin_feature(video)
print(feature)
print(feature.shape)