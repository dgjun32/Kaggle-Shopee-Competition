import os
import numpy
import pandas
import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms

class ShopeeImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transforms, mode = 'train'):
        self.df = df
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image_path = os.path.join(self.cfg['path']['image_dir'], self.df['image'][index])
        img = Image.open(image_path)
        img = self.transforms(img)
        # if test mode, return only image
        if self.mode == 'test':
            return img.float()
        else:
        # else train mode, return image and label for arcface training
            label = torch.tensor(self.df['label_group'][index]).long()
            return img.float(), label

def build_transforms(cfg):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((cfg['model']['img_size'], cfg['model']['img_size'])),
                    transforms.RandomHorizontalFlip(p = 0.5)
    ])
    return transform

class ShopeeTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, mode = 'train'):
        self.df = df
        self.cfg = cfg
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        text = self.df['title'][index]
        if self.mode == 'test':
            return text
        else:
            label = torch.tensor(self.df['label_group'][index]).long()
            return text, label

class ShopeeDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transforms):
        self.df = df
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image_path = os.path.join(self.cfg['path']['image_dir'], self.df['image'][index])
        img = Image.open(image_path)
        img = self.transforms(img)
        text = self.df['title'][index]
        
        return img.float(), text