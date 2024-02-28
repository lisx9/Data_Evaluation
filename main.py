from torchvision.io.video import read_video

vid, _, _ = read_video("D:/lipreading/graduation/dataset/video/20220101100.mp4", output_format="TCHW")
print(vid.shape)