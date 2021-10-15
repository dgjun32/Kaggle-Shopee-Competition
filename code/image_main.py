#basics
import numpy as np
import pandas as pd
import os
import sys
import math

#warnings
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

from config import CFG_VIT
from utils import label_mapper
from image_dataset import ShopeeImageDataset, build_transforms
from image_model import VIT_MODEL

import sklearn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule

if __name__ == "main":
        train = pd.read_csv(CFG_VIT['path']['df'])
        train = label_mapper(train)
        transforms = build_transforms(CFG_VIT)

        # train_test_split
        train_df, val_df = sklearn.model_selection.train_test_split(train,
                                                                test_size = 0.4,
                                                                stratify = train['label_group'], 
                                                                random_state = SEED)
        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        # define datamodule
        traindata = ShopeeImageDataset(train_df, CFG_VIT, transforms, mode = 'train')
        valdata = ShopeeImageDataset(val_df, CFG_VIT, transforms, mode = 'train')
        trainloader = torch.utils.data.DataLoader(traindata, batch_size = CFG_VIT['training']['batch_size'])
        valloader = torch.utils.data.DataLoader(valdata, batch_size = CFG_VIT['training']['batch_size'])
        # define model
        model = VIT_MODEL(CFG_VIT)
        # define callbacks
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
                dirpath = CFG_VIT['path']['output'],
                filename = None,
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                save_last=False,
                )
        logger = TensorBoardLogger(CFG_VIT['model']['name'])
        # Training
        trainer = pl.Trainer(
        logger=logger,
        max_epochs=CFG_VIT['training']['epochs'],
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        gpus = 1)
        trainer.fit(model, trainloader, valloader)

                
