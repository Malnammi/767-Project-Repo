# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on optic disk segmentation 
        using diaretdb1 dataset.
"""

batch_size = 16

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")

import h5py
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import gc
import keras.optimizers
K.set_image_dim_ordering('th')

"""
    Dice coefficient calculation and loss. Credit goes to: github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
    github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/metric.py
"""
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def np_dice_coef(y_true, y_pred):
    smooth = 1.
    tr = y_true.flatten()
    pr = y_pred.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

    
import pandas as pd
"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
kaggle_dir = "G:/767-Project/datasets/kaggle"

train_labels = pd.read_csv(kaggle_dir+"/trainLabels.csv")
test_labels = pd.read_csv(kaggle_dir+"/testLabels.csv", usecols=['image', 'level'])

# get some stats
train_labels.groupby('level').agg('count')
test_labels.groupby('level').agg('count')

# select 400 files to be training for each class
train_files = pd.concat([train_labels[train_labels.level==0][0:400],
          train_labels[train_labels.level==1][0:400],
          train_labels[train_labels.level==2][0:400],
          train_labels[train_labels.level==3][0:400],
          train_labels[train_labels.level==4][0:400]])

# select 400 files to be testing for each class
test_files = pd.concat([test_labels[test_labels.level==0][0:400],
          test_labels[test_labels.level==1][0:400],
          test_labels[test_labels.level==2][0:400],
          test_labels[test_labels.level==3][0:400],
          test_labels[test_labels.level==4][0:400]])
          
# write to csv file
train_files.to_csv("myTrainLabels.csv", index=False)
test_files.to_csv("myTestLabels.csv", index=False)

# load training and test data
X_train = np.array([np.array(Image.open(fname)) for fname in 
                    [kaggle_dir+"/train_128_clahe/"+f+".png" for f in list(train_files['image'])]])
y_train = train_files['level'].as_matrix()
X_test =  np.array([np.array(Image.open(fname)) for fname in 
                    [kaggle_dir+"/test_128_clahe/"+f+".png" for f in list(test_files['image'])]])
y_test = test_files['level'].as_matrix()

# save numpy arrays
np.save(kaggle_dir+"/X_train", X_train)
np.save(kaggle_dir+"/y_train", y_train)
np.save(kaggle_dir+"/X_test", X_test)
np.save(kaggle_dir+"/y_test", y_test)