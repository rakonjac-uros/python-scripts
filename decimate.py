import os
import glob
from PIL import Image 
import os 

os.getcwd()
collection = "/home/uros/NoGPSdata/uki2/"
t2 = open('times_0_0.txt','w')

filenames = glob.glob("images1/*.jpg")
filenames.sort()
#images = [cv2.imread(img) for img in filenames]
i = 0
old = -1
for filename in filenames:
    #print(str(i//10 +1).zfill(9))
    #os.rename(collection + filename, collection + 'image_0/' + str(i+1).zfill(9) + ".png")
    
    
    new = str(i//1 +1)
    if(new == old):
    	continue
    else:
    	t2.write(str(i//1 +1).zfill(9) + '\n')
    	img = Image.open(filename)
    	file_name = str(i//1 +1).zfill(9)
    	print('image_0/'+'{}.png'.format(file_name))
    	img.save('image_0/' + '{}.png'.format(file_name))
    	old = new
    i = i + 1
t2.close()
