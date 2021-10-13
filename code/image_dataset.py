import os
import numpy
import pandas
import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms


# mapping label_group into integer values
def label_mapper(train):
    label_mapper = dict(zip(train['label_group'].unique(), np.arange(len(train['label_group'].unique()))))
    label_mapper_inv = dict(zip(np.arange(len(train['label_group'].unique())), train['label_group'].unique()))
    train['label_group'] = train['label_group'].map(label_mapper)
    return train

class ShopeeDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transforms, mode):
        self.df = df
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image_path = os.path.join(cfg['path']['image_dir'], self.df['image'][index])
        img = Image.open(image_path)
        img = self.transforms(img).float()
        if mode == 'test':
            return img
        else:
            label = self.df['label_group'][index]
            return img, label

def build_transforms(cfg):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((cfg['model']['img_size'], cfg['model']['img_size'])),
                    transforms.RandomHorizontalFlip(p = 0.5)
    ])
    return transform

