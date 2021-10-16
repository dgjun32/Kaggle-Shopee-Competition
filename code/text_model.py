import transformers
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Indonesian PretrainedBert + ArcFace
class IND_BERT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone = eval(cfg['model']['name']).from_pretrained(cfg['model']['weight'])
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
        x = self.embeddings(input['input_ids'], input['token_type_ids'])
        x = self.encoder(x, input['attention_mask'])
        x = x['last_hidden_state'][:,0,:]
        x = self.linear(x)
        x = nn.functional.normalize(x)
        if label is not None:
            arcmargin = self.arcface(x, label)
            return arcmargin
        else:
            return x

    def _step(self, batch):
        tokens, label = batch
        out = self.forward(input = tokens, label = label)
        loss = self.criterion(out, label)
        return out, label, loss
    
    def training_step(self, batch, batch_idx):
        pred, label, loss = self._step(batch)
        tensorboard_log = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_log}
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, label, loss = self._step(batch)
            pred = torch.argmax(pred, axis = 1).reshape(batch[0].shape[0],)
            label = label.reshape(batch[0].shape[0],)
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