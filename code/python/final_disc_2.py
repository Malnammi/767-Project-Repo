# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on optic disk segmentation 
        using diaretdb1 dataset.
"""

batch_size = 32

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu1")

import h5py
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D,AtrousConvolution2D
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
def get_segnet(weights_path=None):
    model = Sequential()
    model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2), input_shape=(1,128,128), activation='relu', 
                            border_mode='same'))
    
    model.add(MaxPooling2D())
    
    model.add(AtrousConvolution2D(32, 3, 3, atrous_rate=(2,2),border_mode='same', activation='relu'))
    model.add(MaxPooling2D())
    
    
    model.add(AtrousConvolution2D(16, 3, 3, atrous_rate=(2,2),border_mode='same', activation='relu'))
    model.add(AtrousConvolution2D(16, 3, 3, atrous_rate=(2,2),border_mode='same', activation='relu'))
    
    model.add(UpSampling2D())    
    model.add(AtrousConvolution2D(32, 3, 3, atrous_rate=(2,2),border_mode='same', activation='relu'))
    
    model.add(UpSampling2D())    
    model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2),border_mode='same', activation='relu'))
                
    model.add(AtrousConvolution2D(1, 1, 1, atrous_rate=(2,2), border_mode='same', activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)    
    
    return model
    

"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
vgg_w = "G:/767-Project/weights/vgg16_weights.h5"
segnet_w = "G:/767-Project/weights/segnet.hdf5"
diaret_dir = "G:/767-Project/datasets/diaretdb1/"

#load training sets
X_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_128/*")])
y_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/disc_128/*")])
X_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/images_128/*")])
y_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/disc_128/*")])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

y_train = y_train.reshape(y_train.shape[0], 128,128, 1)
y_test = y_test.reshape(y_test.shape[0], 128,128, 1)

gc.collect()
gc.collect()
gc.collect()

import cv2
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.5, 0.5, 0]) 
    
    
X_tr = np.zeros((X_train.shape[0], 128, 128, 1))
X_te = np.zeros((X_test.shape[0], 128, 128, 1))

import cv2
for i in range(len(X_train)):
    
    im = X_train[i]
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
    im = rgb2gray(im)      
    X_tr[i,:] = im.reshape(128, 128, 1)  


for i in range(len(X_test)):
    
    im = X_test[i]
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
    im = rgb2gray(im)      
    X_te[i,:] = im.reshape(128, 128, 1)    
    

X_train = X_tr
X_test  = X_te

# input image dimensions
img_rows, img_cols, img_depth = 128, 128, 1


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
seed = 1
#training dataset
#create two datagenerators one for input and out for output masks
train_data_gen_args = dict(rotation_range=180,
                    rescale=1./255,
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


for X_batch, Y_batch in train_generator:
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    plt.imshow(X_batch[0:1,:,:,:].reshape(128,128)/255, cmap='Greys_r')
    a.set_title('X')
    a=fig.add_subplot(1,2,2)
    plt.imshow(Y_batch[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
    a.set_title('Y')
    break

"""############################MODEL AND TRAINING###########################"""
#define model
#model = get_simplenet(input_shape=input_shape)
model = get_segnet()

model.compile(loss=dice_coef_loss, optimizer='adam', 
              metrics=[dice_coef])
              
model.fit_generator(
        train_generator,
        samples_per_epoch=len(X_train),
        nb_epoch=80,
        validation_data=test_generator,
        nb_val_samples=len(X_test))
        
fig = plt.figure()
a=fig.add_subplot(2,3,1)
plt.imshow(X_train[0:1,:,:,:].reshape(128,128)/255, cmap='Greys_r')
a.set_title('train_input_image')
a=fig.add_subplot(2,3,2)
plt.imshow(y_train[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('train_true_label')
a=fig.add_subplot(2,3,3)
y_thresholded = model.predict(X_train[0:1,:,:,:]/255.).reshape(128,128)
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('train_model_thresh')

a=fig.add_subplot(2,3,4)
plt.imshow(X_test[0:1,:,:,:].reshape(128,128)/255, cmap='Greys_r')
a.set_title('test_input_image')
a=fig.add_subplot(2,3,5)
plt.imshow(y_test[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('test_true_label')
a=fig.add_subplot(2,3,6)
y_thresholded = model.predict(X_test[0:1,:,:,:]/255.).reshape(128,128) 
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('test_model_thresh')

plot_feature_maps(get_activations(model, 0, X_train[0:1]/255), 0, 8)