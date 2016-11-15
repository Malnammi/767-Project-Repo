# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:08:01 2016

@author: Moeman
"""
import cv2, glob
import numpy as np
import matplotlib.pyplot as plt

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

im_dir = cv2.imread("G:/767-Project/datasets/diaretdb1/train/images_512/image_001.png")
im_drive = cv2.imread("G:/767-Project/datasets/drive/DRIVE/training/images_512/21_training.png")
im_stare = cv2.imread("G:/767-Project/datasets/stare/vessel_segmentation/image_512/im0001.png")
im_kaggle = cv2.imread("G:/767-Project/datasets/kaggle/train_512/13_left.tiff")

#convert cv2's bgr to rgb ordering
im_dir = cv2.cvtColor(im_dir, cv2.COLOR_BGR2RGB)
im_drive = cv2.cvtColor(im_drive, cv2.COLOR_BGR2RGB)
im_stare = cv2.cvtColor(im_stare, cv2.COLOR_BGR2RGB)
im_kaggle = cv2.cvtColor(im_kaggle, cv2.COLOR_BGR2RGB)

fig = plt.figure()
a=fig.add_subplot(2,2,1)
plt.imshow(im_dir)
a.set_title('diaretb1')

a=fig.add_subplot(2,2,2)
plt.imshow(im_drive)
a.set_title('drive')

a=fig.add_subplot(2,2,3)
plt.imshow(im_stare)
a.set_title('stare')

a=fig.add_subplot(2,2,4)
plt.imshow(im_kaggle)
a.set_title('kaggle')

#Contrast Limited Adaptive Histogram Equalization to normalize colors
im_dir = clahe_color_normalization(im_dir)
im_drive = clahe_color_normalization(im_drive)
im_stare = clahe_color_normalization(im_stare)
im_kaggle = clahe_color_normalization(im_kaggle)

fig = plt.figure()
a=fig.add_subplot(2,2,1)
plt.imshow(im_dir)
a.set_title('diaretb1')

a=fig.add_subplot(2,2,2)
plt.imshow(im_drive)
a.set_title('drive')

a=fig.add_subplot(2,2,3)
plt.imshow(im_stare)
a.set_title('stare')

a=fig.add_subplot(2,2,4)
plt.imshow(im_kaggle)
a.set_title('kaggle')

#subtract local mean color
scale = 300
im_dir = cv2.addWeighted(im_dir, 4,
                         cv2.GaussianBlur(im_dir, (0,0), scale/30), -4,
                         128)
im_drive = cv2.addWeighted(im_drive, 4,
                         cv2.GaussianBlur(im_drive, (0,0), scale/30), -4,
                         128)
im_stare = cv2.addWeighted(im_stare, 4,
                         cv2.GaussianBlur(im_stare, (0,0), scale/30), -4,
                         128)
im_kaggle = cv2.addWeighted(im_kaggle, 4,
                         cv2.GaussianBlur(im_kaggle, (0,0), scale/30), -4,
                         128)
                         
fig = plt.figure()
a=fig.add_subplot(2,2,1)
plt.imshow(im_dir)
a.set_title('diaretb1')

a=fig.add_subplot(2,2,2)
plt.imshow(im_drive)
a.set_title('drive')

a=fig.add_subplot(2,2,3)
plt.imshow(im_stare)
a.set_title('stare')

a=fig.add_subplot(2,2,4)
plt.imshow(im_kaggle)
a.set_title('kaggle')