import cv2
import numpy as np
import glob
 
img_array = []
filenames = glob.glob("/home/uros/Projects/light_detection_testing/RunwayLights4Full/result/*.png")
filenames.sort()
for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
print('for over')

fps = 20
 
out = cv2.VideoWriter('runway_lights_dover_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
print('write over')
out.release()
