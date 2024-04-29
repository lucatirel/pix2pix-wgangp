from PIL import Image
import numpy as np
import os

def is_binary(image):
    unique_values = np.unique(np.array(image))
    print(len(unique_values) == 2)

def check_bin_imgs(path):
    for subdir, dirs, files in os.walk(path):
            for filename in files:
                filepath = subdir + os.sep + filename
                img = Image.open(filepath)
                is_binary(img)
            
data_path = R'C:\Users\luca9\Desktop\Pix2pix WGAN-GP\Code\dataset\clean_new'