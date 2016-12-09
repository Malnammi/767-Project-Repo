# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 00:13:29 2016

@author: Moeman
"""

import cv2
import numpy as np
from PIL import Image


def scaleRadius(img, scale):
    x = img[img.shape[0]/2, :,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)

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
          
          
# load training and test data
X_train = np.array([np.array(Image.open(fname)) for fname in 
                    [kaggle_dir+"/train_128/"+f+".png" for f in list(train_files['image'])]])
y_train = train_files['level'].as_matrix()
X_test =  np.array([np.array(Image.open(fname)) for fname in 
                    [kaggle_dir+"/test_128/"+f+".png" for f in list(test_files['image'])]])
y_test = test_files['level'].as_matrix()

for i in range(len(X_train)):
    
    im = X_train[i]
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), 3), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
    
    X_train[i,:] = im    


for i in range(len(X_test)):
    
    im = X_test[i]
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), 3), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
    
    X_test[i,:] = im   
    
# save numpy arrays
np.save(kaggle_dir+"/X_train1", X_train)
np.save(kaggle_dir+"/y_train1", y_train)
np.save(kaggle_dir+"/X_test1", X_test)
np.save(kaggle_dir+"/y_test1", y_test)
