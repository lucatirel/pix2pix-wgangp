import os
from PIL import Image
from tqdm import tqdm

def convert_image_to_binary(img_path, threshold=128):
    try:
        # Open the image file.
        img = Image.open(img_path).convert('L')
        # Apply a threshold to each pixel.
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
    except Exception as e:
        print(f"Error processing image at {img_path}: {e}")
        return None
    
    return img

def convert_images_in_folder_to_binary(folder_path, threshold=128):
    # Iterate over all files in the given folder.
    for filename in tqdm(sorted(os.listdir(folder_path))):
        # Check if the file is an image.
        if filename.lower().endswith('.png'):
            # Create a full path to the image file.
            img_path = os.path.join(folder_path, filename)
            # Convert the image to binary and save it back to the file.
            binary_img = convert_image_to_binary(img_path, threshold)
            if binary_img is None:
                os.remove(img_path)
                print(f"Deleted corrupt image at {img_path}")
            else:
                binary_img.save(img_path)

# replace 'your_folder_path' with the path to the folder you want to process
convert_images_in_folder_to_binary(R'dataset\clean_new') 
