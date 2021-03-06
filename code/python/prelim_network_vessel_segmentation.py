# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on vessel segmentation using drive
        and stare datasets.
"""
batch_size = 32

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")

from keras.models import Model
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import merge, Permute, Dense, Dropout, Activation, Layer, Flatten, BatchNormalization, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D, UpSampling2D, Deconvolution2D,AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
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

def get_segnet(input_shape, weights_path=None):
        #define model
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    
    enc_l = [
            ZeroPadding2D(padding=(pad,pad), input_shape=input_shape),
            Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(128, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(256, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(512, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
        ]
    
    kernel = 3
    pad = 1
    pool_size = 2
    dec_l = [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
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
    
"""
    Constructs a sementation network structure adapted from: github.com/pradyu1993/segnet
"""
def get_simplenet(input_shape, weights_path=None):
    inputs = Input((3, 128, 128))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    conv6 = Convolution2D(1, 1, 1, activation='relu',border_mode='same')(conv5)
    conv6 = Flatten()(conv6)
    ############
    conv7 = Activation('sigmoid')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='adam', loss=dice_coef_loss,metrics=[dice_coef])

    return model

def get_vgg_16(weights_path=None):
    model = Sequential()
#    model.add(Convolution2D(64, 3, 3, input_shape=(3,128,128), activation='relu', border_mode='same'))
#    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#    
#    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
#    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
#    
#    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same'))
#    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same'))
#    
#    model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same'))
#    model.add(Convolution2D(8, 3, 3, activation='relu', border_mode='same'))
#    
#    model.add(Convolution2D(4, 3, 3, activation='relu', border_mode='same'))
#    model.add(Convolution2D(4, 3, 3, activation='relu', border_mode='same'))
#
#    model.add(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))
#    model.add(Convolution2D(2, 3, 3, activation='relu', border_mode='same'))    
#    
    model.add(Convolution2D(1024, 10, 10, activation='relu', border_mode='same', input_shape=(3,128,128)))
    model.add(Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same'))     
    
    
    if weights_path:
        model.load_weights(weights_path)

    return model
    
#
np.random.seed(7677)  # for reproducibility

#define dataset directories
drive_dir = "G:/767-Project/datasets/drive/DRIVE/"
stare_dir = "G:/767-Project/datasets/stare/vessel_segmentation/"

#load training sets
X_train_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/training/images_128_clahe/*")])
y_train_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/training/labels_128/*")])
X_test_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/test/images_128_clahe/*")])
y_test_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/test/labels_128/*")])

X_train_stare = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/images_128_clahe/*")])
y_train_stare = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/labels_128/*")])
X_test_stare = X_train_stare[0:10,:,:,:]
y_test_stare = y_train_stare[0:10,:,:]
X_train_stare = X_train_stare[10:,:,:]
y_train_stare = y_train_stare[10:,:,:]

#concatenate the two datasets
X_train = np.vstack((X_train_drive,X_train_stare))
y_train = np.vstack((y_train_drive,y_train_stare)) 

X_test = np.vstack((X_test_drive,X_test_stare))
y_test = np.vstack((y_test_drive,y_test_stare))

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

del X_train_drive, X_test_drive, y_train_drive, y_test_drive 
del X_train_stare, X_test_stare, y_train_stare, y_test_stare

# input image dimensions
img_rows, img_cols, img_depth = 128, 128, 3


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_depth, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_depth, img_rows, img_cols)

    y_train = y_train.reshape(y_train.shape[0], 1, img_rows, img_cols)   
    y_test = y_test.reshape(y_test.shape[0], 1, img_rows, img_cols)    
    
    input_shape = (img_depth, img_rows, img_cols)
    output_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_depth)
    
    y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, 1)    
    y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, 1)    
    
    input_shape = (img_rows, img_cols, img_depth)
    output_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


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

#define model
#model = get_segnet(input_shape=input_shape)
#
#model.compile(loss=dice_coef_loss, optimizer='adam', 
#              metrics=[dice_coef])

#model = get_segnet(input_shape)
#               

import keras.optimizers as optimizers

model = get_vgg_16()

model.compile(optimizer='adam',
               loss=dice_coef_loss,
               metrics=[dice_coef])
model.fit_generator(
        train_generator,
        samples_per_epoch=len(X_train),
        nb_epoch=100,
        validation_data=test_generator,
        nb_val_samples=len(X_test))
        
model.fit(X_train/255, y_train/255, batch_size=32,
          nb_epoch=1000,
          validation_data=(X_test/255, y_test/255))
          
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

fig = plt.figure()
a=fig.add_subplot(1,3,1)
plt.imshow(X_train[0:1,:,:,:].reshape(128,128,3)/255)
a.set_title('input_image')
a=fig.add_subplot(1,3,2)
plt.imshow(y_train[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
#a.set_title('true_label')
a=fig.add_subplot(1,3,3)
y_thresholded = model.predict(X_train[0:1,:,:,:]/255.).reshape(128,128)
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
y_thresholded = model.predict(X_test[0:1,:,:,:]/255.).reshape(128,128)
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('model_thresh')