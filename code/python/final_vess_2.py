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
    Function that returns the hidden layer output.
"""
def get_activations(model, layer_num, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], 
                                 model.layers[layer_num].output)
    activations = get_activations([X_batch,0])
    return activations
    
"""
    Plots feature maps.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def plot_feature_maps(feature_maps, start_index, end_index):
    feature_maps = feature_maps.reshape(feature_maps.shape[1],128,128)
    
    end_index = min(end_index, feature_maps.shape[0])
    plt.figure()
    num_plots = end_index - start_index
    n_rows = num_plots//8
    n_cols = 8
    gs = gridspec.GridSpec(n_rows, n_cols, top=1., bottom=0., right=1., left=0., hspace=0.,
        wspace=0.)
    i=0
    for g in gs:
        ax = plt.subplot(g)
        ax.imshow(feature_maps[start_index+i], cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        i=i+1
        
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


def get_vess_segnet(weights_path=None):
    model = Sequential()

    model.add(AtrousConvolution2D(32, 3, 3, input_shape=(1,128,128), border_mode='same', activation='relu'))
    model.add(Convolution2D(16, 2, 2,  border_mode='same', activation='relu'))
    model.add(Convolution2D(8, 2, 2,  border_mode='same', activation='relu'))
    model.add(Convolution2D(4, 3, 3,  border_mode='same', activation='relu'))
    model.add(Convolution2D(2, 3, 3,  border_mode='same', activation='relu'))
    
    model.add(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'))
    
    if weights_path:
        model.load_weights(weights_path)

    return model

#
np.random.seed(7677)  # for reproducibility

#define dataset directories
drive_dir = "G:/767-Project/datasets/drive/DRIVE/"
stare_dir = "G:/767-Project/datasets/stare/vessel_segmentation/"

#load training sets
X_train_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/training/images_128/*")])
y_train_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/training/labels_128/*")])
X_test_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/test/images_128/*")])
y_test_drive = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/test/labels_128/*")])

X_train_stare = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/images_128/*")])
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

X_tr = np.zeros((X_train.shape[0], 128, 128, 1))
X_te = np.zeros((X_test.shape[0], 128, 128, 1))

import cv2
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.5, 0.5, 0]) 
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
    
    im = np.dot(im[...,:3], [0.5, 0.5, 0])
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
    
    im = np.dot(im[...,:3], [0.5, 0.5, 0])
    X_te[i,:] = im.reshape(128, 128, 1)    
    

X_train = X_tr
X_test  = X_te


# input image dimensions
img_rows, img_cols, img_depth = 128, 128, 1


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
train_data_gen_args = dict(rotation_range=180,
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

model = get_vess_segnet()

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
          nb_epoch=100,
          validation_data=(X_test/255, y_test/255))
          
model.fit(X_train/255, y_train.reshape(y_train.shape[0],128*128)/255, batch_size=32,
          nb_epoch=100,
          validation_data=(X_test/255, y_test.reshape(y_test.shape[0],128*128)/255))
      
model.fit(X_train/255, y_train/255, batch_size=32,
          nb_epoch=100,
          validation_data=(X_test/255, y_test/255))
          
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

for X_batch, Y_batch in train_generator:
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    plt.imshow(X_batch[0:1,:,:,:].reshape(128,128)/255, cmap='Greys_r')
    a.set_title('X')
    a=fig.add_subplot(1,2,2)
    plt.imshow(Y_batch[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
    a.set_title('Y')
    break

fig = plt.figure()
a=fig.add_subplot(2,3,1)
plt.imshow(X_train[10:11,:,:,:].reshape(128,128)/255, cmap='Greys_r')
a.set_title('train_input_image')
a=fig.add_subplot(2,3,2)
plt.imshow(y_train[10:11,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('train_true_label')
a=fig.add_subplot(2,3,3)
y_thresholded = model.predict(X_train[10:11,:,:,:]/255.).reshape(128,128) > 0.5
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('train_model_thresh')

a=fig.add_subplot(2,3,4)
plt.imshow(X_test[10:11,:,:,:].reshape(128,128)/255, cmap='Greys_r')
a.set_title('test_input_image')
a=fig.add_subplot(2,3,5)
plt.imshow(y_test[10:11,:,:,:].reshape(128,128), cmap='Greys_r')
a.set_title('test_true_label')
a=fig.add_subplot(2,3,6)
y_thresholded = model.predict(X_test[10:11,:,:,:]/255.).reshape(128,128)  > 0.5
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('test_model_thresh')

fig = plt.figure()
a=fig.add_subplot(1,3,1)
plt.imshow(X_train[0:1,:,:,0].reshape(128,128)/255, cmap='Greys_r')
a.set_title('red_channel')
a=fig.add_subplot(1,3,2)
plt.imshow(X_train[0:1,:,:,1].reshape(128,128)/255, cmap='Greys_r')
a.set_title('green_channel')
a=fig.add_subplot(1,3,3)
plt.imshow(X_train[0:1,:,:,2].reshape(128,128)/255, cmap='Greys_r')
a.set_title('blue_channel')

plot_feature_maps(get_activations(model, 1, X_train[0:1]), 0, 64)

for X_batch, Y_batch in train_generator:
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    plt.imshow(X_batch[0:1,:,:,:].reshape(128,128)/255, cmap='Greys_r')
    a.set_title('X')
    a=fig.add_subplot(1,2,2)
    plt.imshow(Y_batch[0:1,:,:,:].reshape(128,128), cmap='Greys_r')
    a.set_title('Y')
    break