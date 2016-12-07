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
            Convolution2D(64, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            ZeroPadding2D((pad,pad),input_shape=input_shape),
            Convolution2D(64, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
#            Dropout(0.25),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
#            Dropout(0.25),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(16, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(16, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
#            Dropout(0.25)
        ]
    
    kernel = 3
    pad = 1
    pool_size = 2
    dec_l = [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(16, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(16, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
#        Dropout(0.25),
    
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
#        Dropout(0.25),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
#        Dropout(0.25)
    ]
    
    model = Sequential()
    for l in enc_l:
        model.add(l)
    for l in dec_l:
        model.add(l)
    
    if weights_path:
        model.load_weights(weights_path)

    model.add(Convolution2D(1, 1, 1, border_mode='valid'))
    model.add(Activation('sigmoid'))
    return model
    

"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
vgg_w = "G:/767-Project/weights/vgg16_weights.h5"
segnet_w = "G:/767-Project/weights/segnet.hdf5"
diaret_dir = "G:/767-Project/datasets/diaretdb1/"

#load training sets
X_train = np.array([np.array(Image.open(fname.replace('hem_128\\Haemorrhages_', 'images_128_clahe\\image_'))) 
                                            for fname in glob.glob(diaret_dir+"/train/hem_128/*")])
y_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/hem_128/*")])
X_test = np.array([np.array(Image.open(fname.replace('hem_128\\Haemorrhages_', 'images_128_clahe\\image_'))) 
                                        for fname in glob.glob(diaret_dir+"/test/hem_128/*")])
y_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/hem_128/*")])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

y_train = y_train.reshape(y_train.shape[0], 128,128, 1)
y_test = y_test.reshape(y_test.shape[0], 128,128, 1)

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
    
    y_train = y_train.reshape(y_train.shape[0], img_rows,img_cols, 1)
    y_test = y_test.reshape(y_test.shape[0], img_rows,img_cols, 1)
    
    input_shape = (img_rows, img_cols, img_depth)
    output_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


"""################################DATA AUGMENTORS##########################"""
#specify datagenerator for real-time augmentation
datagen = ImageDataGenerator(
        rotation_range=180,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

#training dataset
#create two datagenerators one for input and out for output masks
train_data_gen_args = dict(rotation_range=40,
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_image_datagen = ImageDataGenerator(**train_data_gen_args)
train_mask_datagen = ImageDataGenerator(**train_data_gen_args)

seed = 1
train_image_generator = train_image_datagen.flow(
    X_train,
    batch_size=batch_size,
    seed=seed)

train_mask_generator = train_mask_datagen.flow(
    y_train,
    batch_size=batch_size,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)

#test dataset
#create two datagenerators one for input and out for output masks
test_data_gen_args = dict(rescale=1./255)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)
test_mask_datagen = ImageDataGenerator(**test_data_gen_args)

seed = 1
test_image_generator = test_image_datagen.flow(
    X_test,
    batch_size=batch_size,
    seed=seed)

test_mask_generator = test_mask_datagen.flow(
    y_test,
    batch_size=batch_size,
    seed=seed)

# combine generators into one which yields image and masks
test_generator = zip(test_image_generator, test_mask_generator)


"""############################MODEL AND TRAINING###########################"""
#define model
#model = get_simplenet(input_shape=input_shape)
model_cnn = get_segnet(input_shape=input_shape)

model_cnn.compile(loss=dice_coef_loss, optimizer='adadelta', 
              metrics=[dice_coef])
              
model_cnn.fit_generator(
        train_generator,
        samples_per_epoch=6000,
        nb_epoch=100,
        validation_data=test_generator,
        nb_val_samples=1000)
        
fig = plt.figure()
a=fig.add_subplot(1,3,1)
plt.imshow(X_train[0:1,:,:,:].reshape(128,128,3)/255)
a.set_title('input_image')
a=fig.add_subplot(1,3,2)
plt.imshow(y_train[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('true_label')
a=fig.add_subplot(1,3,3)
y_thresholded = model_cnn.predict(X_train[0:1,:,:,:]/255.).reshape(128,128) > 0.5
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('model_thresh')

fig = plt.figure()
a=fig.add_subplot(1,3,1)
plt.imshow(X_test[0:1,:,:,:].reshape(128,128,3)/255)
a.set_title('input_image')
a=fig.add_subplot(1,3,2)
plt.imshow(y_test[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('true_label')
a=fig.add_subplot(1,3,3)
y_thresholded = model_cnn.predict(X_test[0:1,:,:,:]/255.).reshape(128,128) > 0.1
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('model_thresh')

np_dice_coef(y_train, model_cnn.predict(X_train) > 0.5)