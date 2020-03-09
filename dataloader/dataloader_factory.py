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
    def __init__(self, dataframe, root_dir, folds, transform=None):
        self.df = dataframe[dataframe.fold.isin(folds).reset_index(drop=True)]
        self.root_dir = root_dir
        self.transform = transform
        self.folds = folds

        self.paths = self.df.image_id.values
        self.labels = self.df.values[:,2:5]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.paths[idx]
        img_path = f'{self.root_dir}{img_name}.png'
        
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (128,128))

        if self.transform is not None:
          img = self.transform(image=img)['image']
        
        # img = np.expand_dims(img, axis=0)
        # img = np.concatenate([img, img, img])
        img = np.rollaxis(img, -1, 0)

        labels = np.array(self.labels[idx]).astype(np.long)
        return [img, labels]

def get_loader(dataframe, rootdir, folds, batch_size=128, workers=0, shuffle=True, transform=None):
    dataset = ImageDataset(dataframe, rootdir, folds, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loader
