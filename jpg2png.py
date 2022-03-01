from PIL import Image 
import os 

path = r'/home/uros/NoGPSArducam/noGPSimagesC1/image_0_jpg/'
save_path = '/home/uros/NoGPSArducam/noGPSimagesC1/image_0/'
i = 0
for file in os.listdir(path): 
    if file.endswith(".jpg"): 
        print(i)
        img = Image.open(path+file)
        file_name, file_ext = os.path.splitext(file)
        #file_name = file_name.replace('20_20210802_','')
        #file_name = file_name.replace('_','')
        img.save(save_path+'{}.png'.format(file_name))
        i = i + 1 
