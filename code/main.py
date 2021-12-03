import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import sklearn
import argparse
from sklearn.model_selection import train_test_split

from config import CFG_VIT, CFG_BERT
from image_model import VIT_MODEL
from text_model import IND_BERT
from datasets import ShopeeImageDataset, ShopeeTextDataset, build_transforms
from utils import label_mapper

print('Succesfully installed libraries')

def main():
    # parsing
    parser = argparse.ArgumentParser(description='DataEncoder Training')
    parser.add_argument('--model_type', choices = ['image', 'text'], dest='model_type')
    parser.add_argument('--gpu', choices = ['cuda:0', 'cuda:1'], dest='gpu_id')
    parser.add_argument('--seed', type=int, default=42, dest='seed')
    args = parser.parse_args()
    print(args)
    # set device
    torch.cuda.set_device(args.gpu_id)
    # load dataset
    train = pd.read_csv(CFG_VIT['path']['df'])
    train = label_mapper(train)
    train_df, val_df = sklearn.model_selection.train_test_split(train,
                                                                test_size = 0.4,
                                                                stratify = train['label_group'], 
                                                                random_state = args.seed)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    # set configuration & model & dataloader
    if args.model_type == 'image':
        # cfg
        cfg = CFG_VIT
        # model
        model = VIT_MODEL(cfg)
        model.cuda()
        # data
        transforms = build_transforms(cfg)
        traindata = ShopeeImageDataset(train_df, cfg, transforms, mode = 'train')
        valdata = ShopeeImageDataset(val_df, cfg, transforms, mode = 'train')
        trainloader = torch.utils.data.DataLoader(traindata, batch_size = CFG_VIT['training']['batch_size'])
        valloader = torch.utils.data.DataLoader(valdata, batch_size = CFG_VIT['training']['batch_size'])
    else:
        # cfg
        cfg = CFG_BERT
        # model
        model = IND_BERT(cfg)
        model.cuda()
        # data
        traindata = ShopeeTextDataset(train_df, cfg, mode = 'train')
        valdata = ShopeeTextDataset(val_df, cfg, mode = 'train')
        trainloader = torch.utils.data.DataLoader(traindata, batch_size = cfg['training']['batch_size'])
        valloader = torch.utils.data.DataLoader(valdata, batch_size = cfg['training']['batch_size'])
    print('Load model with backbone : {}'.format(cfg['model']['name']))
    print('Set Dataloader')
    # set optimizer
    if args.model_type == 'text':
        optimizer_arcface = eval(cfg['training']['optim'])(model.arcface.parameters(),
                                                         lr = cfg['training']['arcface_lr'])
        para = list()
        for p in model.encoder.parameters():
            para.append(p)
        for p in model.linear.parameters():
            para.append(p)
        optimizer_backbone = eval(cfg['training']['optim'])(para,
                                                            lr = cfg['training']['backbone_lr'])                                  
    else:
        optimizer_arcface = eval(cfg['training']['optim'])(model.arcface.parameters(),
                                                             lr = cfg['training']['arcface_lr'])
        optimizer_backbone = eval(cfg['training']['optim'])(model.backbone.parameters(),
                                                            lr = cfg['training']['backbone_lr'])
    # set scheduler
    scheduler_arcface = OneCycleLR(optimizer_arcface,
                                max_lr = cfg['training']['arcface_lr'],
                                epochs = cfg['training']['epochs'],
                                steps_per_epoch = len(trainloader),
                                anneal_strategy = 'cos',
                                pct_start = 0.1)         
    scheduler_backbone = OneCycleLR(optimizer_backbone,
                                max_lr = cfg['training']['backbone_lr'],
                                epochs = cfg['training']['epochs'],
                                steps_per_epoch = len(trainloader),
                                anneal_strategy = 'cos',
                                pct_start = 0.1) 
    # start training
    total_steps = 0
    for epoch in range(cfg['training']['epochs']):
        # Train
        print('Training....')
        epoch_loss = 0
        epoch_steps = 0
        for step, batch in enumerate(trainloader):
            optimizer_arcface.zero_grad()
            optimizer_backbone.zero_grad()
            data, label = batch
            # forward propagate
            pred = model(data, label)
            loss = nn.CrossEntropyLoss()(pred, label.cuda())
            # backward propagate
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        cfg['training']['grad_clip'])
            # update weights
            optimizer_arcface.step()
            optimizer_backbone.step()
            # update lr
            scheduler_arcface.step()
            scheduler_backbone.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1
            # verbosity
            if (step+1) % 100 == 0:
                print("{} epoch {}th step | Training loss: {:.2e}  |lr_1: {:.2e} lr_2: {:.2e}".format(epoch+1,
                                                                                        step+1,
                                                                                        epoch_loss/epoch_steps,
                                                                                        optimizer_arcface.param_groups[0]['lr'],
                                                                                        optimizer_backbone.param_groups[0]['lr']))
        # checkpoint
        torch.save(model.state_dict(), os.path.join(cfg['path']['output'], '{}_encoder_{}steps'.format(args.model_type, total_steps))+'.pth')
        # Validate
        print('Validating....')
        val_pred = []
        val_label = []
        for i,batch in enumerate(valloader):
            text, label = batch
            with torch.no_grad():
                pred = model(input = text)
                pred = torch.topk(pred, 10, dim = 1).indices # batch_size x 10
                label = label.reshape(pred.shape[0],)
                val_pred.append(pred)
                val_label.append(label)
        val_pred = torch.cat(val_pred, dim=0).cpu()
        val_label = torch.cat(val_label, dim=0).cpu()
        ans = 0
        for i in range(val_pred.shape[0]):
            if val_label[i] in val_pred[i]:
                ans += 1
        acc = ans / val_pred.shape[0]
        print('Top-10 acc on {} epoch : {:.2e}%'.format(epoch+1, acc*100))

if __name__ == '__main__':
    main()

