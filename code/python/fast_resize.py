"""
Fast image resize script: DRD@Kaggle

__author__ : Abhishek Thakur
"""

#Credit goes to author who made this available at: www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/12803/fast-image-resize-script

import os
import glob
from joblib import Parallel, delayed

in_dir = 'G:\\767 Project\\datasets\\kaggle\\train_orig'
out_dir = 'G:\\767 Project\\datasets\\kaggle\\train_512'
IMAGE_SIZE = 512

from PIL import Image, ImageChops
JPEG_FILES = glob.glob(in_dir+'*.jpeg')
def convert(img_file):
    im = Image.open(img_file)
    im.resize((IMAGE_SIZE,IMAGE_SIZE)).save(out_dir + os.path.basename(img_file), 'JPEG')
    #im.thumbnail((IMAGE_SIZE,IMAGE_SIZE,Image.ANTIALIAS))

Parallel(n_jobs=5, verbose=10)(delayed(convert)(f) for f in JPEG_FILES)
	