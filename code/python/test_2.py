batch_size = 64

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu1")

import h5py
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D, AtrousConvolution2D
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def plot_feature_maps(feature_maps, start_index, end_index, n_cols):
    feature_maps = feature_maps.reshape(feature_maps.shape[1],
                                        feature_maps.shape[2]
                                        ,feature_maps.shape[3])
    
    end_index = min(end_index, feature_maps.shape[0])
    
    num_plots = end_index - start_index
    n_rows = num_plots//n_cols
    plt.figure()
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

"""
    get disc segmentation
"""
def get_disc_segnet(weights_path=None):
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

"""
    get vessel segmentation
"""
def get_vess_segnet(weights_path=None):
    model = Sequential()

    model.add(Convolution2D(8, 3, 3, input_shape=(1,128,128), border_mode='same', activation='relu'))
    model.add(Convolution2D(16, 3, 3,  border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 3, 3,  border_mode='same', activation='relu'))
    model.add(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'))
    
    if weights_path:
        model.load_weights(weights_path)

    return model


"""
    Function to transform rgb of images to grayscale
"""
def rgb2gray(X_orig):
    X_tr = np.zeros((X_orig.shape[0], 128, 128, 1))
    
    for i in range(len(X_orig)):        
        rgb = X_orig[i,:].reshape(128,128,3)        
        X_tr[i,:] = np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).reshape(128,128,1)

    return X_tr.reshape(X_tr.shape[0], 1, 128, 128)


"""
    Optic Disc image conversion.
"""
def disc_conversion(X_orig):    
    X_tr = np.zeros((X_orig.shape[0], 128, 128, 1))
    
    import cv2
    for i in range(len(X_orig)):
        
        im = X_orig[i].reshape(128,128,3)
        
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
    
    return X_tr.reshape((X_orig.shape[0], 1, 128, 128))

"""
    vessel image conversion.
"""
def vess_conversion(X_orig):    
    X_tr = np.zeros((X_orig.shape[0], 128, 128, 1))
    
    import cv2
    for i in range(len(X_orig)):        
        im = X_orig[i].reshape(128,128,3)
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
    
    return X_tr.reshape((X_orig.shape[0], 1, 128, 128))
 
"""
    Kaggle image conversion.
"""
def kaggle_conversion(X_orig, depth):    
    X_tr = np.zeros((X_orig.shape[0], 128, 128, depth))
    
    import cv2
    for i in range(len(X_orig)):
        
        im = X_orig[i].reshape(128,128,depth)
        
        scale = 64
    
        im = cv2.addWeighted(im, 4,
                             cv2.GaussianBlur(im, (0,0), scale/30), -4,
                             128)
                             
        
        b=np.zeros(im.shape)
        cv2.circle(b, (64, 64),
                   int(scale*0.9), (1,1,1), -1, 8, 0)
        
        im = im * b + 128*(1-b)  
#        im = np.dot(im[...,:3], [0.299, 0.587, 0.114])    
        X_tr[i,:] = im.reshape(128, 128, depth)  
    
    return X_tr.reshape((X_orig.shape[0], depth, 128, 128))
    
    
"""
    Function to print random sample of images and their resulting model
    prediction
"""
def plot_model_sample1(X, model, sample_size, m_type):
    indices = np.random.choice(len(X), sample_size)
    
    X_tr = None
    if m_type == 'disc':
        X_tr = disc_conversion(X[indices,:])/255
    else:
        X_tr = vess_conversion(X[indices,:])/255
    
    pred = model.predict(X_tr)

    plt.figure()
    
    gs = gridspec.GridSpec(3, sample_size, top=1., bottom=0., right=1., left=0., hspace=0.,
        wspace=0.)
    i=0
    
    pred = np.concatenate((X_tr, pred), axis=0)
    X_orig = X[indices,:]
    for g in gs:
        ax = plt.subplot(g)
        if i < sample_size:
            ax.imshow(X_orig[i].reshape(128,128,3)/255, cmap='Greys_r')
        else:
            ax.imshow(pred[i-sample_size].reshape(128,128), cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        i=i+1
    
    
"""#################################START OF CODE###########################"""
#K.set_image_dim_ordering('th')
np.random.seed(7677)  # for reproducibility

#define dataset directories
kaggle_dir = "G:/767-Project/datasets/kaggle"

#load training sets
X_train = np.load(kaggle_dir+"/X_train2.npy")
y_train = np.load(kaggle_dir+"/y_train2.npy")
X_test = np.load(kaggle_dir+"/X_test2.npy")
y_test = np.load(kaggle_dir+"/y_test2.npy")

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
model_disc = get_disc_segnet('weights/disc4.h5')

model_disc.compile(loss=dice_coef_loss, optimizer='adam', 
              metrics=[dice_coef])        

#generate disc segmented kaggle images
train_disc = model_disc.predict(disc_conversion(X_train)/255)
test_disc =  model_disc.predict(disc_conversion(X_test)/255)

#load vess model

#convert kaggle images
X_train = kaggle_conversion(X_train, 3)/255
X_test = kaggle_conversion(X_test, 3)/255

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
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#training dataset
#create two datagenerators one for input and out for output masks
train_data_gen_args = dict(rotation_range=180,
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
test_data_gen_args = dict(rescale=1)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)

seed = 1
test_generator = test_image_datagen.flow(
    X_test,
    y_test,
    batch_size=batch_size,
    seed=seed)
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
from keras.optimizers import SGD
model.load_weights('G:/767-Project/weights/vgg16_weights.h5')
input_shape
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='adam', 
              metrics=['accuracy'])
  
model.fit(X_train, y_train, batch_size=512,
          nb_epoch=30,
          validation_data=(X_test, y_test))
