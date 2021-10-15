import numpy as np
import pandas as pd

# mapping label_group into integer values
def label_mapper(train):
    label_mapper = dict(zip(train['label_group'].unique(), np.arange(len(train['label_group'].unique()))))
    label_mapper_inv = dict(zip(np.arange(len(train['label_group'].unique())), train['label_group'].unique()))
    train['label_group'] = train['label_group'].map(label_mapper)
    return train