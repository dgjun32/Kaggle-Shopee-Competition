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

#warnings
import warnings

# arcface layer implementation
def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return training

class ArcFace(tf.keras.layers.Layer):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
    
    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):
        
        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shapes):
        embedding_shape, label_shape = input_shapes
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')
    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output

# Image Encoder with EfficientNetB3 backbone
class ImageEncoder(Layer):
    def __init__(self, input_shape, num_classes, scale, margin):
        super(ImageEncoder, self).__init__()
        self.base_model = keras.applications.EfficientNetB3(include_top = False, weights = 'imagenet', input_shape = input_shape)
        for layer in self.base_model.layers:
            layer.trainable = False
        self.gap = GlobalAveragePooling2D()
        self.batchnorm = BatchNormalization()
        self.l2norm = Lambda(lambda x: K.l2_normalize(x,axis=1))
        self.arcface = ArcFace(num_classes = num_classes, s = scale, m = margin)
        self.softmax = Activation('softmax') 
    def call(self, input_image, input_label):
        x = self.base_model(input_image)
        x = self.gap(x)
        x = self.batchnorm(x)
        x = self.l2norm(x)
        x = self.arcface([x, input_label])
        output = self.softmax(x)
        return output