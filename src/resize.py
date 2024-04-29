import os
from PIL import Image

# Source directory
src_dir = R"C:\Users\luca9\Desktop\Pix2pix WGAN-GP\Code\dataset\noisy"

# Output directory
out_dir = src_dir
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Iterate over all files in source directory
for filename in os.listdir(src_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # add any other image formats if needed
        image_path = os.path.join(src_dir, filename)

        # Open image
        img = Image.open(image_path)

        # Check if either dimension is less than 1024
        if img.size[0] < 1024 or img.size[1] < 1024:
            # Maintain aspect ratio
            smaller_dim = min(img.size)
            scale = 1024 / smaller_dim

            # Compute new dimensions
            new_size = tuple([round(x * scale) for x in img.size])

            # Resize image
            resized_img = img.resize(new_size, Image.ANTIALIAS)

            # Save resized image to output directory
            out_path = os.path.join(out_dir, filename)
            resized_img.save(out_path)

        else:
            print(f"Image {filename} is already larger than 1024x1024, skipping...")

print("Resizing complete!")
