# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:40:43 2016

@author: Moeman
"""

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

def clahe_color_normalization(source_rgb):
    """
    Adapted and credit goes to author at: http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
    
    """
    #-----Converting image to LAB Color model----------------------------------- 
    lab = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2LAB)
    
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final
    
diaret_dir = "G:/767-Project/datasets/diaretdb1/"
kaggle_dir = "G:/767-Project/datasets/kaggle/sample/"
drive_dir = "G:/767-Project/datasets/drive/DRIVE/training/images_128/"
stare_dir = "G:/767-Project/datasets/stare/vessel_segmentation/images_128/"


images = np.zeros((20, 128, 128,3))
images[0:5,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/*")])[0:5,:]
images[5:10,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/*")])[0:5,:]
images[10:15,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_128/*")])[0:5,:]
images[15:20,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(kaggle_dir+"/*")])[0:5,:]


plt.figure()
gs = gridspec.GridSpec(4, 5, top=1., bottom=0., right=1., left=0., hspace=0.,
    wspace=0.)
i=0
for g in gs:
    ax = plt.subplot(g)
    ax.imshow(images[i]/255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    i=i+1
    


images = np.zeros((20, 128, 128,3), dtype='uint8')
images[0:5,:] = np.array([np.array(cv2.imread(fname)) for fname in glob.glob(drive_dir+"/*")])[0:5,:]
images[5:10,:] = np.array([np.array(cv2.imread(fname)) for fname in glob.glob(stare_dir+"/*")])[0:5,:]
images[10:15,:] = np.array([np.array(cv2.imread(fname)) for fname in glob.glob(diaret_dir+"/train/images_128/*")])[0:5,:]
images[15:20,:] = np.array([np.array(cv2.imread(fname)) for fname in glob.glob(kaggle_dir+"/*")])[0:5,:]

for i in range(len(images)):    
    im = images[i]
    
    im = clahe_color_normalization(im)
    #subtract local mean
    scale = 300
    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)  
    images[i,:] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


plt.figure()
gs = gridspec.GridSpec(4, 5, top=1., bottom=0., right=1., left=0., hspace=0.,
    wspace=0.)
i=0
for g in gs:
    ax = plt.subplot(g)
    ax.imshow(images[i]/255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    i=i+1
    
images = np.zeros((20, 128, 128,3))
images[0:5,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/*")])[0:5,:]
images[5:10,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/*")])[0:5,:]
images[10:15,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_128/*")])[0:5,:]
images[15:20,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(kaggle_dir+"/*")])[0:5,:]

import cv2
for i in range(len(images)):    
    im = images[i].reshape(128,128,3)
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
#        im = np.dot(im[...,:3], [0.299, 0.587, 0.114])    
    images[i,:] = im.reshape(128, 128, 3)  


plt.figure()
gs = gridspec.GridSpec(4, 5, top=1., bottom=0., right=1., left=0., hspace=0.,
    wspace=0.)
i=0
for g in gs:
    ax = plt.subplot(g)
    ax.imshow(images[i]/255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    i=i+1
images = np.zeros((20, 128, 128,3))
images[0:5,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(drive_dir+"/*")])[0:5,:]
images[5:10,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(stare_dir+"/*")])[0:5,:]
images[10:15,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(diaret_dir+"/train/images_128/*")])[0:5,:]
images[15:20,:] = np.array([np.array(Image.open(fname)) for fname in glob.glob(kaggle_dir+"/*")])[0:5,:]

import cv2
images_g = np.zeros((20, 128, 128))
for i in range(len(images)):    
    im = images[i].reshape(128,128,3)
    
    scale = 64

    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)
                         
    
    b=np.zeros(im.shape)
    cv2.circle(b, (64, 64),
               int(scale*0.9), (1,1,1), -1, 8, 0)
    
    im = im * b + 128*(1-b)  
    im = np.dot(im[...,:3], [0.5, 0.5, 0])    
    images_g[i,:] = im.reshape(128, 128)  


plt.figure()
gs = gridspec.GridSpec(4, 5, top=1., bottom=0., right=1., left=0., hspace=0.,
    wspace=0.)
i=0
for g in gs:
    ax = plt.subplot(g)
    ax.imshow(images_g[i]/255, cmap='Greys_r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    i=i+1
    