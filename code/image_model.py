#basics
import numpy as np
import pandas as pd
import os
import sys
import math

#modeling
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule

#image preprocessing and input pipeline
from PIL import Image

#warnings
import warnings

# ArcFace Module
class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

# Vision Transformer based feature extractor
class VIT_MODEL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg['model']['name']).from_pretrained(cfg['model']['weight'])
        self.backbone = self.backbone.vit
        # freezing backbone weight
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.arcface = ArcMarginProduct(in_feature = 768,
                                           out_feature = cfg['model']['num_classes'],
                                           s = cfg['model']['scale'],
                                           m = cfg['model']['margin'])
    def forward(self, input, label = None):
        x = self.backbone(input.cuda())
        x = x['last_hidden_state'][:,0,:]
        x = nn.functional.normalize(x)
        if label is not None:
            arcmargin = self.arcface(x, label.cuda())
            return arcmargin
        else:
            return x

    