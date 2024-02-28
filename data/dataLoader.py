import os
import torch, torchvision
import torch.utils.data as data
from data.transforms import VideoTransform
from detector.video_process import VideoProcess
from detector.detector import LandmarksDetector
from torchvision.models.video import s3d, S3D_Weights

class MyDataset(data.Dataset):
    def __init__(self, data_folder, speed_rate=1, transform=True, convert_gray=True):
        self.data_folder = data_folder
        self.filenames = []
        self.labels = []
        self.video_process = VideoProcess(convert_gray=convert_gray)
        self.transform = transform
        self.video_transform = VideoTransform(speed_rate=speed_rate)
        video_path = os.path.join(data_folder, "video")
        text_path = os.path.join(data_folder, "text")
        video_names = os.listdir(video_path)
        text_names = os.listdir(text_path)
        for i in video_names:
            self.filenames.append(os.path.join(video_path, i))
        for i in text_names:
            f = open(os.path.join(text_path, i), encoding="utf-8")
            data = f.readlines()
            str = ''.join(data)
            self.labels.append(str)

    def __getitem__(self, item):
        label = self.labels[item]
        video_path = self.filenames[item]
        video = self.load_video(video_path)
        landmark_detector = LandmarksDetector()
        landmarks = landmark_detector(video_path)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return self.video_transform(video) if self.transform else video, label

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit='sec')[0].numpy()

    def get_s3d_feature(self, vid):
        weights = S3D_Weights.KINETICS400_V1
        model = s3d(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        batch = preprocess(vid).unsqueeze(0)
        feature = model(batch).squeeze(0)
        return feature