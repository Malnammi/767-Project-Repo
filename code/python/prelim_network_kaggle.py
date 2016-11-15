# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 01:29:48 2016

@author: Moeman
Purpose: Preliminary convnetwork on kaggle dataset for testing.
"""

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
import h5py
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imsave

