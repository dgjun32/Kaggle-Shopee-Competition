import numpy
import pandas
import torch
import torch.nn as nn
import sklearn
import argparse
from sklearn.model_selection import train_test_split

from config import CFG_VIT, CFG_BERT
from image_model import VIT_MODEL
from text_model import IND_BERT
from datasets import ShopeeImageDataset, ShopeeTextDataset, build_transforms
from utils import label_mapper

def main():
    # parsing
    parser = argparse.ArgumentParser(description='DataEncoder Training')
    parser.add_argument('--model_type', choices = ['image', 'text'], dest='model_type')
    parser.add_argument('--gpu', choices = ['cuda:0', 'cuda:1'], dest='gpu_id')
    parser.add_argument('--seed', type=int, default=42, dest='seed')
    args = parser.parse_args()

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
    
    # set optimizer and scheduler
    optimizer = eval(cfg['training']['optimizer'])(model.parameters(),
                                                lr = cfg['training']['learning_rate'])
    scheduler = eval(cfg['training']['lr_scheduler'])(optimizer)
    
    # start training
    total_steps = 0
    for _ in trange(cfg['training']['epochs'], desc='Epochs'):
        # Train
        print('Training....')
        epoch_loss = 0
        epoch_steps = 0
        tqdm_train = tqdm(trainloader, desc='Training')
        for step, batch in enumerate(tqdm_train):
            optimizer.zero_grad()
            data, label = batch
            # forward propagate
            pred = model(data, label)
            loss = nn.CrossEntropyLoss()(pred, label)
            # backward propagate
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1
            # verbosity
            tqdm_train.desc = "Training loss: {:.2e} lr: {:.2e}".format(epoch_loss/epoch_steps, scheduler.get_lr()[0])
        # checkpoint
        torch.save(model.state_dict(), os.path.join(cfg['path']['output'], '{}_encoder_{}steps'.format(args.model_type, total_steps)))
        # Validate
        print('Validating....')
        tqdm_val = tqdm(valloader, desc='Validating')
        val_pred = []
        val_label = []
        for batch in tqdm_val:
            text, label = batch
            with torch.no_grad():
                pred = model(text = text)
                pred = torch.argmax(pred, axis=1).reshape(pred.shape[0],)
                label = label.reshape(pred.shape[0],)
                pred.to('cpu')
                label.to('cpu')
                val_pred.append(pred)
                val_label.append(label)
        val_pred = torch.cat(val_pred, dim=0)
        val_label = torch.cat(val_label, dim=0)
        acc = (val_pred == val_label).sum() / len(val_pred)
        print('Validation acc: {:.2e}'.format(acc))

if __name__ == '__main__':
    main()

