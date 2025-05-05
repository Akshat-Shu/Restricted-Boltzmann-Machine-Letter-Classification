from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
from Model.config import Config
import torch

class LetterImgDataset(Dataset):
    def __init__(self, img_directory, label_file, use_cuda=False):
        labels = {}
        csv = pd.read_csv(label_file)
        labels = csv.to_dict()['label']

        self.labels = labels
        self.img_directory = img_directory
        self.list_items = csv.to_dict()['image']

        self.cfg = Config()

        self.label_keys = self.cfg.key_labels

        self.use_cuda = use_cuda

 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.list_items[idx]
        # print(file_name)
        file_path = os.path.join(self.img_directory, file_name)
        label = self.labels[idx]

        img = self.load_image(file_path)

        one_hot_label = self.one_hot_encode(label)

        if self.use_cuda:
            img = img.cuda()
            one_hot_label = one_hot_label.cuda()

        return img, one_hot_label


    def one_hot_encode(self, label):
        one_hot = torch.zeros(len(self.label_keys))
        if self.use_cuda:
            one_hot = one_hot.cuda()
        one_hot[self.label_keys[label]] = 1.0
        return one_hot
    
    @staticmethod
    def one_hot(n_C, labels):
        one_hot = torch.zeros(len(labels), n_C, dtype=torch.float32, device=labels.device)
        for i, label in enumerate(labels):
            one_hot[i][label] = 1.0
        return one_hot
    
    def one_hot_decode(self, one_hot):
        item = torch.argmax(one_hot, dim=0).item()
        label = 0
        for k, v in self.label_keys.items():
            if v == item:
                label = k
                break
        return label

    def load_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found at {file_path}")
        img = img[self.cfg.crop_x1:self.cfg.crop_x2+1, self.cfg.crop_y1:self.cfg.crop_y2+1]
        img = cv2.resize(img, (self.cfg.im_size, self.cfg.im_size))
        binary_img = 1 - img / 255.0
        return torch.tensor(binary_img.flatten(), dtype=torch.float32)
    


def get_dataloader(img_directory, label_file, batch_size=32, shuffle=True, use_cuda=False):
    dataset = LetterImgDataset(img_directory, label_file, use_cuda=use_cuda)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset