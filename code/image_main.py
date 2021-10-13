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

from config import CFG_VIT as cfg
from image_dataset import label_mapper, ShopeeImageDataset
from image_model import VIT_MODEL

if __name__ == "__main__"
