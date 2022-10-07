import os
import shutil

img_list = os.listdir('ground_truth')

for img in img_list:
    if(not os.path.exists(os.path.join('./masks',img[:4]))):
        os.mkdir(os.path.join('./masks/',img[:4]))
    shutil.copyfile(os.path.join('ground_truth/',img), os.path.join('./masks/',img[:4],img[5:]))
