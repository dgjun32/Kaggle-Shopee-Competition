from utils import label_mapper
from text_dataset import ShopeeTextDataset
from image_model import IND_BERT
from config import CFG_BERT

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
    # train_test_split
    train_df, val_df = sklearn.model_selection.train_test_split(train, test_size = 0.4, stratify = train['label_group'], random_state = SEED)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()

    # define datamodule
    traindata = ShopeeTextDataset(train_df, CFG_BERT, mode = 'train')
    valdata = ShopeeTextDataset(val_df, CFG_BERT, mode = 'train')
    trainloader = torch.utils.data.DataLoader(traindata, batch_size = CFG_BERT['training']['batch_size'])
    valloader = torch.utils.data.DataLoader(valdata, batch_size = CFG_BERT['training']['batch_size'])

    # define model
    model = IND_BERT(CFG_BERT)

    # define callbacks
    earystopping = EarlyStopping(monitor="val_loss")
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath = CFG_BERT['path']['output'],
        filename = None,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
        )
    logger = TensorBoardLogger(CFG_BERT['model']['name'])

    # train
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=CFG_BERT['training']['epochs'],
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        gpus = 1)
    trainer.fit(model, trainloader, valloader)