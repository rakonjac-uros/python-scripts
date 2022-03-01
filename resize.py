#!/usr/bin/python

from PIL import Image
import os, sys

path = "images1/"
dirs = os.listdir( path )

def resize():
    i = 0
    for item in dirs:
        if os.path.isfile(path+item):
            i = i + 1
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((480,480), Image.LANCZOS)
            imResize.save('images/'+ item, 'JPEG', quality=90)
            print(i)

resize()
