#basics
import numpy as np
import pandas as pd
import os
import sys
import sklearn
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#warnings
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

from config import config_arcface as cfg
from dataset import ArcFaceImageGenerator
from image_model import ImageEncoder

if __name__ == "__main__"

    # dataset
    traingen = ArcFaceImageGenerator(batch_size = cfg.training.batch_size,
                                     target_size = [cfg.model.img_size, cfg.model.img_size],
                                    x_col = 'img_path', y_col = 'label_group', df = train_df[:20000])

    valgen = ArcFaceImageGenerator(batch_size = cfg.training.batch_size,
                                     target_size = [cfg.model.img_size, cfg.model.img_size],
                                    x_col = 'img_path', y_col = 'label_group', df = train_df[20000:])
    
    # image model
    input_img = Input(shape = (cfg.model.img_size, cfg.model.img_size,3), dtype = tf.float32)
    input_label = Input(shape = (1,), dtype = tf.int32)
    output = ImageEncoder(input_shape = (cfg.model.img_size, cfg.model.img_size,3),
                          num_classes = 11014, scale = 50, margin = 0.4)(input_img, input_label)
    image_model = tf.keras.models.Model([input_img, input_label], output)

    # model compile
    lr_schedule = eval(cfg.model.training.lr_schedule.name)(
            initial_learning_rate = cfg.model.training.lr_schedule.learning_rate,
            first_decay_steps = cfg.model.training.lr_schedule.first_decay_steps,
            t_mul=cfg.model.training.lr_schedule.t_mul,
            m_mul=cfg.model.training.lr_schedule.m_mul,
            alpha=cfg.model.training.lr_schedule.alpha)
    image_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule))
    # model checkpoint
    model_check = keras.callbacks.ModelCheckpoint(    
            filepath = cfg.checkpoint.path,
            save_weights_only = False)

    # train
    history = image_model.fit_generator(traingen, epochs = 5, callbacks = [model_check])