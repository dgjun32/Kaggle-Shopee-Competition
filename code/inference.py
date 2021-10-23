import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cuml
from cuml.neigbors import NearestNeighbors

from image_model import VIT_MODEL
from text_model import IND_BERT
from metric import f1_score

class Matching:
    def __init__(self, dataloader, cfg_img, cfg_text):
        self.cfg_img = cfg_img
        self.cfg_text = cfg_text
        self.best_threshold = None
        img_model = VIT_MODEL(cfg_img).load_state_dict(cfg['path']['model']).eval()
        text_model = IND_BERT(cfg_text).load_state_dict(cfg['path']['model']).eval()
        self.embeddings = self._compute_representations(img_model, text_model, dataloader, tokenizer)
        del img_model, text_model
        gc.collect()

    def _compute_representations(self, img_model, text_model, dataloader):
        img_reps, text_reps = [], []
        img_model.cuda()
        text_model.cuda()
        for i, batch in enumerate(dataloader):
            imgs, texts = batch
            imgs.cuda()
            texts.cuda()
            img_rep = img_model(imgs)
            text_rep = text_model(texts)
            img_reps.append(img_rep)
            text_reps.append(text_rep)
        return torch.cat([torch.cat(img_reps, dim=0), torch.cat(text_reps, dim=0)], dim = 1)

    def _get_neighbors_cv(self, df, thresholds, k):
        '''
        df : dataframe containing product id
        thresholds : List[float] ; candidates for optimal threshold
        k : maximum number of products to be considered as same product
        '''
        model = NearestNeighbors(n_neighbors = k)
        model.fit(self.embeddings)
        distances, indices = model.kneighbors(self.embeddings)
        
        scores = []
        for threshold in cv_threshold_range:
            predictions = []
            for k in range(self.embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k,idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['matching'] = predictions
            df['f1_score'] = f1_score(df['match'], df['matching'])
            score = df['f1_score'].mean()
            scores.append(score)
            print(f'score : {score} when threshold : {threshold}')
        
        thresholds_scores = pd.DataFrame({'thresholds': list(cv_threshold_range), 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'best score is {best_score}, when threshold is {best_threshold}')
        self.best_threshold = best_threshold
    
    def _get_neighbors(self, df, best_threshold, k):
        model = NearestNeighbors(n_neighbors = k)
        model.fit(self.embeddings)
        distances, indices = model.kneighbors(self.embeddings)
        
        # if executed cv
        if best_threshold is None:
            best_threshold = self.best_threshold
        
        predictions = []
        for k in tqdm(range(self.embeddings.shape[0])):
            idx = np.where(distances[k,] < best_threshold)[0]
            ids = indices[k,idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)
        df['matching'] = predictions
        return df