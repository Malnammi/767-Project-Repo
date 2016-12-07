# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:19:35 2016

@author: Moeman
Purpose: CLAHE color normalize images of all training sets
"""

import os
import cv2, glob
import numpy as np
from multiprocessing.pool import Pool

N_PROC = 10

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

def convert_and_write(img_file, output_dir):
    im = cv2.imread(img_file) 
    #Contrast Limited Adaptive Histogram Equalization to normalize colors
    im = clahe_color_normalization(im)
    #subtract local mean
    scale = 300
    im = cv2.addWeighted(im, 4,
                         cv2.GaussianBlur(im, (0,0), scale/30), -4,
                         128)
    #write to output_dir
    fname = img_file.split('\\',1)[-1]
    cv2.imwrite(output_dir+fname, im)

def process(args):
    fun, arg = args
    fname, output_dir = arg
    fun(fname, output_dir)


#directories
drive_dir = "G:/767-Project/datasets/drive/DRIVE"
stare_dir = "G:/767-Project/datasets/stare/vessel_segmentation"
diaret_dir = "G:/767-Project/datasets/diaretdb1"
kaggle_dir = "G:/767-Project/datasets/kaggle"


##drive dataset
image_files = glob.glob(drive_dir+"/training/images_128/*")
output_dir = "G:/767-Project/datasets/drive/DRIVE/training/images_128_clahe/"
for f in image_files:
    convert_and_write(f, output_dir)
    
image_files = glob.glob(drive_dir+"/test/images_128/*")
output_dir = "G:/767-Project/datasets/drive/DRIVE/test/images_128_clahe/"
for f in image_files:
    convert_and_write(f, output_dir)
    
    
##stare dataset
image_files = glob.glob(stare_dir+"/images_128/*")
output_dir = "G:/767-Project/datasets/stare/vessel_segmentation/images_128_clahe/"
for f in image_files:
    convert_and_write(f, output_dir)


##diaretdb1 dataset
image_files = glob.glob(diaret_dir+"/train/images_128/*")
output_dir = "G:/767-Project/datasets/diaretdb1/train/images_128_clahe/"
for f in image_files:
    convert_and_write(f, output_dir)
    
image_files = glob.glob(diaret_dir+"/test/images_128/*")
output_dir = "G:/767-Project/datasets/diaretdb1/test/images_128_clahe/"
for f in image_files:
    convert_and_write(f, output_dir)

def main():
    #kaggle dataset
    image_files = glob.glob(kaggle_dir+"/train_128/*")
    output_dir = "G:/767-Project/datasets/kaggle/train_128_clahe/"
    
    n = len(image_files)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 20000
    batches = n // batchsize + 1
    pool = Pool(N_PROC)
    
    args = []
    
    for f in image_files:
        args.append((convert_and_write, (f, output_dir)))
    
    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])
    
    pool.close()
    
    print('done')
    
    image_files = glob.glob(kaggle_dir+"/test_128/*")
    output_dir = "G:/767-Project/datasets/kaggle/test_128_clahe/"
        
    n = len(image_files)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 20000
    batches = n // batchsize + 1
    pool = Pool(N_PROC)
    
    args = []
    
    for f in image_files:
        args.append((convert_and_write, (f, output_dir)))
    
    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])
    
    pool.close()
    
    print('done')	

if __name__ == '__main__':
    main()