import os
import glob
os.getcwd()
collection = "/home/uros/Desktop/NvRec_0/test_ts2/"
#t2 = open('/home/uros/NoGPSArducam/noGPSimagesC1/times.txt','w')

filenames = glob.glob("/home/uros/Desktop/NvRec_0/test_ts2/*.png")
filenames.sort()
#images = [cv2.imread(img) for img in filenames]
i = 1
for filename in filenames:
    print(filename)
    
    os.rename(filename, collection + str(i).zfill(3) + ".png")
    
    #t2.write(str(i).zfill(3) + '\n')
    i = i + 1
#t2.close()
