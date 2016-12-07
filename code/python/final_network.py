# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:58:05 2016

@author: Moeman
Purpose: perliminary network for training on optic disk segmentation 
        using diaretdb1 dataset.
"""

batch_size = 64

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
def plot_feature_maps(feature_maps, start_index, end_index):
    feature_maps = feature_maps.reshape(feature_maps.shape[1],
                                        feature_maps.shape[2],
                                        feature_maps.shape[3])
    
    fig = plt.figure()
    end_index = min(end_index, feature_maps.shape[0])
    
    num_plots = end_index - start_index
    n_rows = num_plots/5
    n_cols = 5
    
    for i in range(num_plots):     
        a=fig.add_subplot(n_rows,n_cols,i+1)
        plt.imshow(feature_maps[start_index+i], cmap='Greys_r')
        a.set_title('fmap'+str(start_index+i))
        

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
    get disc segmentation
"""
def get_disc_segnet(input_shape, weights_path=None):
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
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(32, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(16, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(16, kernel, kernel, border_mode='valid'),
            BatchNormalization(mode=1),
            Activation('relu'),
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
    
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='valid'),
        BatchNormalization(mode=1),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='valid'),
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
    model.add(Activation('sigmoid'))
    return model
    

"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
kaggle_dir = "G:/767-Project/datasets/kaggle"

#load training sets
X_train = np.load(kaggle_dir+"/X_train.npy")
y_train = np.load(kaggle_dir+"/y_train.npy")
X_test = np.load(kaggle_dir+"/X_test.npy")
y_test = np.load(kaggle_dir+"/y_test.npy")

# convert class vectors to binary class matrices
nb_classes=5
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

gc.collect()
gc.collect()
gc.collect()


#now reshape appropriately
# input image dimensions
img_rows, img_cols, img_depth = 128, 128, 3


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_depth, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_depth, img_rows, img_cols)
    
    input_shape = (img_depth, img_rows, img_cols)
    output_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_depth)
    
    input_shape = (img_rows, img_cols, img_depth)
    output_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


#load disc model
model_disc = get_disc_segnet(input_shape=input_shape)

model_disc.compile(loss=dice_coef_loss, optimizer='adam', 
              metrics=[dice_coef])
              
model_disc.load_weights('weights/disc_seg.h5')
        

#generate disc segmented kaggle images
train_disc = model_disc.predict(X_train)
test_disc =  model_disc.predict(X_test)

#concatenate with kaggle images
X_train = np.concatenate((X_train, train_disc), axis=1)
X_test = np.concatenate((X_test, test_disc), axis=1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

img_rows, img_cols, img_depth = 128, 128, 4
input_shape = (img_depth, img_rows, img_cols)

"""################################DATA AUGMENTORS##########################"""
#specify datagenerator for real-time augmentation
datagen = ImageDataGenerator(
        rotation_range=180,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#training dataset
#create two datagenerators one for input and out for output masks
train_data_gen_args = dict(rotation_range=180,
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_image_datagen = ImageDataGenerator(**train_data_gen_args)

seed = 1
train_generator = train_image_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size,
    seed=seed)


#test dataset
#create two datagenerators one for input and out for output masks
test_data_gen_args = dict(rescale=1./255)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)

seed = 1
test_generator = test_image_datagen.flow(
    X_test,
    y_test,
    batch_size=batch_size,
    seed=seed)


"""############################MODEL AND TRAINING###########################"""
"""
    adapted design from: http://jeffreydf.github.io/diabetic-retinopathy-detection/
"""

#define model
model = Sequential()

model.add(Convolution2D(32,7,7, subsample=(2,2), input_shape=input_shape))
model.add(Activation(LeakyReLU(0.5)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(Convolution2D(32,3,3, subsample=(1,1)))        
model.add(Activation(LeakyReLU(0.5)))
model.add(Convolution2D(32,3,3, subsample=(1,1)))        
model.add(Activation(LeakyReLU(0.5)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(Convolution2D(64,3,3, subsample=(1,1)))     
model.add(Activation(LeakyReLU(0.5)))   
model.add(Convolution2D(64,3,3, subsample=(1,1)))
model.add(Activation(LeakyReLU(0.5)))        
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

#model.add(Convolution2D(128,3,3, subsample=(1,1)))        
#model.add(Convolution2D(128,3,3, subsample=(1,1)))      
#model.add(Convolution2D(128,3,3, subsample=(1,1)))        
#model.add(Convolution2D(128,3,3, subsample=(1,1)))    
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

#model.add(Convolution2D(256,3,3, subsample=(1,1)))        
#model.add(Convolution2D(256,3,3, subsample=(1,1)))      
#model.add(Convolution2D(256,3,3, subsample=(1,1)))        
#model.add(Convolution2D(256,3,3, subsample=(1,1)))    
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

          
model.fit_generator(
        train_generator,
        samples_per_epoch=len(X_train),
        nb_epoch=100,
        validation_data=test_generator,
        nb_val_samples=len(X_test))
  
model.fit(X_train/255, y_train, batch_size=32,
          nb_epoch=1,
          validation_data=(X_test/255, y_test))
          
        
fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(X_train[0:1,:,:,:].reshape(128,128,3)/255)
a.set_title('input_image')
a=fig.add_subplot(1,2,2)
y_thresholded = model.predict(X_train[0:1,:,:,:]/255.).reshape(128,128) > 0.5
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('model_thresh')

fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(X_test[0:1,:,:,:].reshape(128,128,3)/255)
a.set_title('input_image')
a=fig.add_subplot(1,2,2)
y_thresholded = model.predict(X_test[0:1,:,:,:]/255.).reshape(128,128) > 0.5
plt.imshow(y_thresholded, cmap='Greys_r')
a.set_title('model_thresh')

feature_maps = get_activations(model, 1, X_train[0:1])
plot_feature_maps(feature_maps, 0, 10)

quadratic_weighted_kappa(np.argmax(y_train, axis=1), np.argmax(model.predict(X_train), axis=1))

quadratic_weighted_kappa(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
