import numpy as np
import cv2
from scipy import ndimage


import os

import sys
import argparse

a = argparse.ArgumentParser()
a.add_argument("--img", help="path to image")
a.add_argument("--pathOut", nargs = '?', const="rotated.png", type=str, help="path-to-images-folder/")

args = a.parse_args()

print(args)
img = args.img


image1 = cv2.imread(img)
gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize = 3)

canimg = cv2.Canny(gray, 50, 200)
lines= cv2.HoughLines(canimg, 1, np.pi/180.0, 250, np.array([]))
#lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image1,(x1,y1),(x2,y2),(0,0,255),2)
    
print(theta)
print(rho)
#cv2.imshow("image",image1)
#cv2.imshow("Edges",edges)
  
img_rotated = ndimage.rotate(image1, 180*theta/3.1415926)
#cv2.imshow("Rotated", img_rotated)

cv2.imwrite(str(args.pathOut),img_rotated)
