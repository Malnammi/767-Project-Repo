# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on optic disk segmentation 
        using diaretdb1 dataset.
"""

batch_size = 32

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")

import h5py
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
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

"""
    Constructs vgg-16 structure adapted from: gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
"""
def get_vgg_16(input_shape, weights_path=None, nb_classes=None):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(Flatten())
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(nb_classes, activation='softmax'))
   
    if weights_path:
        model.load_weights(weights_path)
        
    model.add(ZeroPadding2D((1,1))) 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))    
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(UpSampling2D(size=(2,2)))    
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(Convolution2D(1, 1, 1, border_mode='valid'))

    return model

"""
    Constructs batchnormalized vgg-16 structure adapted from: gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
"""
def get_bnorm_vgg_16(input_shape, weights_path=None, nb_classes=None):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(Flatten())
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(nb_classes, activation='softmax'))
   
    if weights_path:
        model.load_weights(weights_path)
        
    model.add(ZeroPadding2D((1,1))) 
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3)) 
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    
    model.add(UpSampling2D(size=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    
    model.add(UpSampling2D(size=(2,2)))    
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3))
    BatchNormalization(mode=1)
    model.add(Activation('relu'))
    
    model.add(Convolution2D(1, 1, 1, border_mode='valid'))

    return model
    
"""
    Constructs a sementation network structure adapted from: github.com/pradyu1993/segnet
"""
def get_segnet(input_shape, weights_path=None):
        #define model
    kernel = 3
    pad = 1
    pool_size = 2
    
    enc_l = [
            ZeroPadding2D((pad,pad),input_shape=input_shape),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(64, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(128, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
        ]
    
    kernel = 3
    pad = 1
    pool_size = 2
    dec_l = [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
    
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
    
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
    ]
    
    model = Sequential()
    for l in enc_l:
        model.add(l)
    for l in dec_l:
        model.add(l)
    
    if weights_path:
        model.load_weights(weights_path)

    model.add(Convolution2D(1, 1, 1, border_mode='valid'))
    return model
    
"""
    Simple conv net.
"""
def get_simplenet(input_shape, weights_path=None):
        #define model
    kernel = 3
    pad = 1
    pool_size = 2
    
    enc_l = [
            ZeroPadding2D((pad,pad),input_shape=input_shape),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('sigmoid'),
            Dropout(0.5),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            ZeroPadding2D((pad,pad),input_shape=input_shape),
            Convolution2D(16, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('sigmoid'),
            Dropout(0.5),
            MaxPooling2D(pool_size=(pool_size, pool_size))
        ]
    
    kernel = 3
    pad = 1
    pool_size = 2
    dec_l = [
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(16, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('sigmoid'),
        Dropout(0.5),
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('sigmoid'),
        Dropout(0.5)
    ]
    
    model = Sequential()
    for l in enc_l:
        model.add(l)
    for l in dec_l:
        model.add(l)
    
    if weights_path:
        model.load_weights(weights_path)

    model.add(Convolution2D(1, 1, 1, border_mode='valid'))
    return model

"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
vgg_w = "G:/767-Project/weights/vgg16_weights.h5"
segnet_w = "G:/767-Project/weights/segnet.hdf5"
diaret_dir = "G:/767-Project/datasets/diaretdb1/"

#load training sets
X_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_128_clahe/*")])
y_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/disc_128/*")])
X_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/images_128_clahe/*")])
y_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/disc_128/*")])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

y_train = y_train.reshape(y_train.shape[0], 128*128, 1)
y_test = y_test.reshape(y_test.shape[0], 128*128, 1)

gc.collect()
gc.collect()
gc.collect()

#now reshape appropriately
# input image dimensions
img_rows, img_cols, img_depth = 128, 128, 3


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_depth, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_depth, img_rows, img_cols)

    y_train = y_train.reshape(y_train.shape[0], 1, img_rows,img_cols)
    y_test = y_test.reshape(y_test.shape[0], 1, img_rows,img_cols)
    
    input_shape = (img_depth, img_rows, img_cols)
    output_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_depth)
    
    y_train = y_train.reshape(y_train.shape[0], img_rows*img_cols, 1)
    y_test = y_test.reshape(y_test.shape[0], img_rows*img_cols, 1)
    
    input_shape = (img_rows, img_cols, img_depth)
    output_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')



"""############################MODEL AND TRAINING###########################"""
#define model
model = get_simplenet(input_shape=input_shape)
#model.add(Reshape((1,128*128)))
model.add(Activation('sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])
model.fit(X_train/255., y_train/255., batch_size=32,
          validation_data=(X_test/255., y_test/255.),
          nb_epoch=1000)
#model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])
model.fit(X_train/255., y_train/255., batch_size=32,
          validation_data=(X_test/255., y_test/255.),
          nb_epoch=1000)
          

score = model.evaluate(X_test/255., y_test/255., verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

x = model.predict(X_train[0:1,:,:,:]/255.)
x = x[0,:,:].reshape(128,128)
plt.imshow(x>0.5, cmap='Greys_r')
plt.figure()
plt.imshow(y_train[0:1,:,:].reshape(128,128), cmap='Greys_r')


x = model.predict(X_test[0:1,:,:,:]/255.)
x = x[0,:,:].reshape(128,128)
plt.imshow(x > 0.5, cmap='Greys_r')
plt.figure()
plt.imshow(y_test[0:1,:,:].reshape(128,128), cmap='Greys_r')