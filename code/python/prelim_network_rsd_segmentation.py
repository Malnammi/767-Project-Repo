# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on red small dots segmentation 
        using diaretdb1 dataset.
"""

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import gc

#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
diaret_dir = "G:/767-Project/datasets/diaretdb1/"

#load training sets
X_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_512_clahe/*")])
y_train = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/rsd_512/*")])
X_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/images_512_clahe/*")])
y_test = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/test/rsd_512/*")])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

gc.collect()
gc.collect()
gc.collect()

#now reshape appropriately
batch_size = 10
# input image dimensions
img_rows, img_cols, img_depth = 512, 512, 3


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
train_data_gen_args = dict(rotation_range=180,
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
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
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

model = Sequential()

model.add(Convolution2D(32, 5, 5,
                        border_mode='same',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(1, 1, 1, border_mode='same'))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        samples_per_epoch=50,
        nb_epoch=50,
        validation_data=test_generator,
        nb_val_samples=50)
        
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])