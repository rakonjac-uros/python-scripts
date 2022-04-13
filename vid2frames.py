import sys
import argparse
import os

import cv2
print(cv2.   __version__)

def extractImages(pathIn, pathOut, wait_ms):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*wait_ms))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame ('+str(count) +'): ' + str(success))
        if success:
        	cv2.imwrite( os.path.join(pathOut, "frame" + str(count).zfill(4)+ ".png"), image)     # save frame as JPEG file
        	count = count + 1
        else:
            break

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path-to-images-folder/")
    a.add_argument("--wait_ms", help="save frame every wait_ms ms")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut, int(args.wait_ms))
    print('Done! Goodbye!')
