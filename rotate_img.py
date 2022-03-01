import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from matplotlib import cm
import os
import time

import sys
import argparse


def binarizeImage(RGB_image):

  image = rgb2gray(RGB_image)
  threshold = threshold_otsu(image)
  bina_image = image < threshold
  
  return bina_image
  
def findEdges(bina_image):
  
  image_edges = sobel(bina_image)
  return image_edges
  

def findTiltAngle(image_edges):
  
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])
  
  if (angle < 0):
    
    angle = angle + 90
    
  else:
    
    angle = angle - 90
   
  return angle
  

def rotateImage(RGB_image, angle):
  
  return rotate(RGB_image, angle)
  


a = argparse.ArgumentParser()
a.add_argument("--img", help="path to image")
a.add_argument("--pathOut", nargs = '?', const="./", type=str, help="path-to-images-folder/")
args = a.parse_args()
print(args)
img = args.img
start = time.time()
image = io.imread(img)
bina_image = binarizeImage(image)
image_edges = findEdges(bina_image)
angle = findTiltAngle(image_edges)
res_img = rotateImage(io.imread(img), angle)
print("rotation time: ", time.time() - start )
io.imsave(str(args.pathOut)+str(os.path.basename(img)[:-4])+"_rot.png",res_img)
print("rotation + save time: ", time.time() - start )
