import os
import glob
os.getcwd()
collection = "/home/uros/Projects/light_detection_testing/RunwayLights4/"

filenames = glob.glob("/home/uros/Projects/light_detection_testing/RunwayLights4/*.png")
i = 1
for filename in filenames:
    print(filename)
    f = os.path.basename(filename)
    #os.rename(filename, collection + "frame" +str(f[5:-4]).zfill(3) + ".png")
    os.rename(filename, collection + "frame" +f)
    i = i + 1

