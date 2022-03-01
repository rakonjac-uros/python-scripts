import os
import glob
os.getcwd()
collection = "/home/uros/Desktop/NvRec_0/test_ts2/"

filenames = glob.glob("/home/uros/Desktop/NvRec_0/test_ts2/*.png")
i = 1
for filename in filenames:
    print(filename)
    f = os.path.basename(filename)
    os.rename(filename, collection + str(f[5:-4]).zfill(3) + ".png")
    i = i + 1

