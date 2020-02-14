import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd

stats = (0.0692, 0.2051)

class ImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img_name = self.df[idx][0]
        img_name = self.df.index[idx]
        img_path = f'{self.root_dir}{img_name}.png'
        
        if self.transform is not None:
          # img = cv2.imread(img_path)
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          img = self.transform(img)
          img = np.concatenate([img, img, img])
          # res = self.transform(img, severity=5)
          # img = np.rollaxis(res, -1, 0)
          # img = res['image'].astype(np.float32)
        else:
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          # Normalize
          img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]
          # img = cv2.resize(img, (64, 64))
          img = np.expand_dims(img, axis=0)
          img = np.concatenate([img, img, img])
        
        # labels = np.array(self.df[idx][1:-1])
        labels = np.array(self.df.iloc[idx])
        return [img, labels]

def get_loader(dataframe, rootdir, batch_size=128, workers=0, shuffle=True, training=True, transform=None):
    dataset = ImageDataset(dataframe, rootdir, training, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loader
