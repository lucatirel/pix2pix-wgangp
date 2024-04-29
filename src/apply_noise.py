import os
from augraphy import *
from augraphy.default.pipeline import *
from PIL import Image
import numpy as np
from tqdm import tqdm


def add_noise_to_image(img_path, output_path, noise_model):
    img = Image.open(img_path)
    img_np = np.array(img).astype('uint8')
    image_augmented = noise_model(img_np)
    image_augmented = Image.fromarray(image_augmented * 255) 
    image_augmented.save(output_path)

def process_images_in_folder(folder_path, output_folder):
    noise_model = BadPhotoCopy(noise_type=1,
                                   noise_iteration=(2,3),
                                   noise_size=(2,3),
                                   noise_sparsity=(0.35,0.35),
                                   noise_concentration=(0.3,0.3),
                                   blur_noise=-1,
                                   blur_noise_kernel=(5, 5),
                                   wave_pattern=0,
                                   edge_effect=0)
    
    
    # Iterate over all files in the given folder.
    # sorted_files = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))

    for filename in tqdm(os.listdir(folder_path)):
        # Check if the file is an image.
        if filename.lower().endswith('.png'):
            # Create a full path to the image file.
            img_path = os.path.join(folder_path, filename)
            # Create a full path for the output file.
            output_path = os.path.join(output_folder, filename)
            # Convert the image to binary and add noise.
            add_noise_to_image(img_path, output_path, noise_model)


# replace 'your_folder_path' and 'your_output_folder' with the path to the folder you want to process and the output folder
process_images_in_folder(R'dataset/clean_new', R'dataset/noisy_new')
