import os
from PIL import Image
from tqdm import tqdm

def convert_to_png(path):
    # Get the total number of files for tqdm progress bar
    for filename in tqdm(sorted(os.listdir(path))):
        filepath = os.path.join(path, filename)
        
        if filepath.endswith(".jpg") and not os.path.exists(filepath[:-4] + '.png'):
            # print(f"Converting: {filename} -> {filename[:-4] + '.png'}")
            try:
                im = Image.open(filepath)
                rgb_im = im.convert('RGB')
                rgb_im.save(filepath[:-4] + '.png', "PNG")
            except Exception as e:
                print(f"Error converting image {filepath}. Error: {str(e)}")
            

convert_to_png(R'C:\Users\luca9\Desktop\Pix2pix WGAN-GP\Code\dataset\clean_new')  
