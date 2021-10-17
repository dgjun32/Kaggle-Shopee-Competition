import transformers
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

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

class IND_BERT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone = eval(cfg['model']['name']).from_pretrained(cfg['model']['weight'])
        self.tokenizer = transformers.BertTokenizer.from_pretrained(cfg['model']['weight'])
        self.embeddings = backbone.embeddings
        self.encoder = backbone.encoder
        self.linear = backbone.pooler.dense
        # freezing backbone weight
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False
        self.arcface = ArcMarginProduct(in_feature = 768,
                                           out_feature = cfg['model']['num_classes'],
                                           s = cfg['model']['scale'],
                                           m = cfg['model']['margin'])
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input, label = None):
        # tokenization
        input = self.tokenizer.batch_encode_plus(input, max_length = self.cfg['model']['max_length'],
                                                 padding = 'max_length', truncation = True)
        # forward propagation
        x = self.embeddings(torch.tensor(input['input_ids']).cuda(), torch.tensor(input['token_type_ids']).cuda())
        x = self.encoder(x)
        x = x['last_hidden_state'][:,0,:]
        x = self.linear(x)
        x = nn.functional.normalize(x)
        if label is not None:
            arcmargin = self.arcface(x, label)
            return arcmargin
        else:
            return x

    def _step(self, batch):
        text, label = batch
        out = self.forward(input = text, label = label)
        loss = self.criterion(out, label)
        return out, label, loss
    
    def training_step(self, batch, batch_idx):
        pred, label, loss = self._step(batch)
        tensorboard_log = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_log}
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, label, loss = self._step(batch)
            pred = torch.argmax(pred, axis = 1).reshape(pred.shape[0],)
            label = label.reshape(pred.shape[0],)
        return {'val_pred': pred, 'val_label': label, 'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred = torch.cat([x['val_pred'] for x in outputs])
        label = torch.cat([x['val_label'] for x in outputs])
        pred.to('cpu')
        label.to('cpu')
        acc = (pred == label).sum() / len(pred)
        print(f"Epoch {self.current_epoch} avg_loss:{avg_loss} acc:{acc}")
        self.log('val_loss', avg_loss)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss,
                'val_acc': acc,
                'log': tensorboard_logs}
            
    def configure_optimizers(self):
        optimizer = eval(self.cfg['training']['optim'])(self.parameters(),
                                                        lr = self.cfg['training']['lr_schedule']['learning_rate'])
        schedule = eval(self.cfg['training']['lr_schedule']['name'])(optimizer = optimizer,
                                                                     T_max = self.cfg['training']['epochs'])
        return [optimizer], [schedule]