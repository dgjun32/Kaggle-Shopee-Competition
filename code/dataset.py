#basics
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
import math

#modeling
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

#image preprocessing and input pipeline
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
import cv2

#warnings
import warnings

# DataGenerator for Metric Learning with Triplet loss
def dataframe_generator(df, BATCH_SIZE, STEPS, K = 10):
    dataframe = pd.DataFrame()
    images = []
    labels = []
    for step in range(STEPS):
        labels_in_batch = np.random.choice(df['label_group'].unique(), BATCH_SIZE, replace = False)
        images_batch = []
        labels_batch = np.repeat(labels_in_batch, K)
        for label in labels_in_batch:
            imgs = np.random.choice(df.loc[df['label_group'] == label, 'img_path'], K, replace = True)
            for img in imgs:
                images_batch.append(img)
        
        images.append(images_batch)
        labels.append(labels_batch)
    images = np.concatenate(images)
    labels =np.concatenate(labels)
    dataframe['images'] = images
    dataframe['labels'] = labels
    return dataframe

class TripletImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col, batch_size = 10, K = 10, target_size = (512, 512), shuffle = False):
        '''
        batch_size : BATCH_SIZE
        K : number of candidate images from duplicates 
        '''
        self.df = df
        self.len_df = len(df)
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.K = K
        
        self.generator = ImageDataGenerator(rescale = 1./255,
                                            shear_range=0.5,
                                            zoom_range=0.5,
                                            rotation_range = 60,
                                            zca_whitening = True,
                                            horizontal_flip=True)
        self.df_generator = self.generator.flow_from_dataframe(dataframe=self.df, 
                                                            x_col = self.x_col,
                                                            y_col = self.y_col,
                                                            target_size = self.target_size,
                                                            color_mode = 'rgb',
                                                            class_mode = 'raw',
                                                            batch_size = self.batch_size * self.K,
                                                            shuffle = self.shuffle,
                                                            seed = 42)
        self.df_generator.reset = False
        self.batch_index = 0
        self.on_epoch_end()
        
    def __len__(self):
        self.steps =  int(self.len_df / (self.batch_size * self.K))
        return self.steps
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.len_df)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        else:
            val = self.df.values
            M, n = self.len_df//self.K, 2
            val = val.reshape(M,-1,n)[np.random.permutation(M)].reshape(-1,n)
            self.df = pd.DataFrame(val, columns = ['images', 'labels'])
    
    def __getitem__(self, index):
        # Generate data
        if self.batch_index >= self.steps - 1:
            self.batch_index = 0
        else:
            self.batch_index = self.batch_index + 1
        images, labels = self.df_generator.__getitem__(self.batch_index)
        return images, labels

# DataGenerator for utilizing ArcFace layer
class ArcFaceImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, target_size, x_col, y_col, df, shuffle = True):
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.df = df
        self.len_df = len(self.df)
        self.indexes = np.arange(self.len_df)
        self.num_classes = len(np.unique(self.df[self.y_col]))
        self.generator = ImageDataGenerator(rescale = 1./255,
                                            shear_range=0.5,
                                            zoom_range=0.5,
                                            rotation_range = 60,
                                            zca_whitening = True,
                                            horizontal_flip=True)
        self.datagen = self.generator.flow_from_dataframe(dataframe=self.df, 
                                                            x_col = self.x_col,
                                                            y_col = self.y_col,
                                                            target_size = self.target_size,
                                                            color_mode = 'rgb',
                                                            class_mode = 'raw',
                                                            batch_size = self.batch_size,
                                                            shuffle = self.shuffle,
                                                            seed = 42)
    def __len__(self):
        self.steps = int(self.len_df / self.batch_size)
        return self.steps
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        images, sparse_labels = self.datagen.__getitem__(index)
        return ([images, sparse_labels], sparse_labels)