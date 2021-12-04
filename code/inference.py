import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import argparse

from image_model import VIT_MODEL
from text_model import IND_BERT
from datasets import ShopeeDataset, build_transforms
from utils import label_mapper
from metric import f1_score
from config import CFG_VIT as cfg_img
from config import CFG_BERT as cfg_text

class Matching:
    def __init__(self, dataloader, cfg_img, cfg_text):
        self.cfg_img = cfg_img
        self.cfg_text = cfg_text
        self.best_threshold = None
        self.img_model = VIT_MODEL(cfg_img)
        self.img_model.load_state_dict(torch.load(cfg_img['path']['model']))
        self.text_model = IND_BERT(cfg_text)
        self.text_model.load_state_dict(torch.load(cfg_text['path']['model']))
        self.dataloader = dataloader
        self._compute_representations()
        self._set_multimodal_embeddings()
        del self.img_model, self.text_model
        del self.img_embeddings, self.text_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    def _compute_representations(self):
        print('Start Computing Embeddings')
        img_reps, text_reps = [], []
        self.img_model.cuda()
        self.text_model.cuda()
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                imgs, texts = batch
                img_rep = self.img_model(imgs)
                text_rep = self.text_model(texts)
                img_reps.append(img_rep)
                text_reps.append(text_rep)
                if (i+1) % 50 == 0:
                    print('{}th img and texts are processed'.format(32*(i+1))) 
        self.img_embeddings = torch.cat(img_reps, dim=0)
        self.text_embeddings = torch.cat(text_reps, dim=0)
        print('Finished Computing Embeddings')
    
    def _set_multimodal_embeddings(self):
        # concatenating image embedding and text embedding
        embeddings = torch.stack([self.img_embeddings, self.text_embeddings], dim=1).norm(dim=-1, keepdim=True)
        self.embeddings = embeddings.cpu().numpy()
    # CV for selecting optimal threshold with validation set
    def match_cv(self, df, threshold_li):
        # ground truth matchinig for CV
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['match'] = df.label_group.map(tmp)
        # matching cv with many threshold score
        best_t = None
        best_f1 = 0.0
        scores = []
        sim_mat = cosine_similarity(self.embeddings, self.embeddings)
        for t in threshold_li:
            pred = []
            for r in range(sim_mat.shape[0]):
                idx = np.where(sim_mat[r] > t)[0]
                ids = ' '.join(df['posting_id'].iloc[idx].values)
                pred.append(ids)
            df['pred_matching'] = pred
            df['f1_score'] = f1_score(df['match'], df['matching'])
            f1 = df['f1_score'].mean()
            scores.append(f1)
            print('threshold : {} | f1_score : {}'.format(t, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        print(f'best score is {best_f1}, when threshold is {best_t}')
    # For inference only
    def match(self, df, threshold):
        sim_mat = cosine_similarity(self.embeddings, self.embeddings)
        posting_id = df['posting_id'].values
        pred = []
        for r in range(sim_mat.shape[0]):
            idx = np.where(sim_mat[r] > t)[0]
            ids = ' '.join(df['posting_id'].iloc[idx].values)
            pred.append(ids)
        sub = pd.DataFrame({'posting_id':posting_id, 'matches':pred})
        return sub

if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Matching Prediction')
    parser.add_argument('--model_type', choices = ['image', 'text'], dest='model_type')
    parser.add_argument('--gpu', choices = ['cuda:0', 'cuda:1'], dest='gpu_id')
    parser.add_argument('--seed', type=int, default=42, dest='seed'),
    parser.add_argument('--cv', type=bool, default=True, dest='cv')
    parser.add_argument('--threshold', type=float, default=0, dest='threshold')
    args = parser.parse_args()
    print(args)

    # set device
    torch.cuda.set_device(args.gpu_id)
    
    # load dataset
    train = pd.read_csv(cfg_img['path']['df'])
    test_df = pd.read_csv(cfg_img['path']['test_df'])
    train = label_mapper(train)
    train_df, val_df = train_test_split(train,
                                        test_size = 0.4,
                                        stratify = train['label_group'], 
                                        random_state = args.seed)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    transforms = build_transforms(cfg_img, mode = 'test')
    if args.cv:
        # set dataloader
        dataset = ShopeeDataset(val_df, cfg_img, transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32)
        # initiate matching engine
        matching_engine = Matching(dataloader, cfg_img, cfg_text)
        # search best threshold
        thresholds = list(range(0.1,1,0.05))
        matching_engine.match_cv(val_df, thresholds)
    else:
        dataset = ShopeeDataset(test_df, cfg_img, transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32)
        # initiate matching engine
        matching_engine = Matching(dataloader, cfg_img, cfg_text)
        # inference
        sub = matching_engine.match(test_df, args.threshold)
        sub.to_csv(os.path.join(cfg_img['path']['submission'], 'submission.csv'))

