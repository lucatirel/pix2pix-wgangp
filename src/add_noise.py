import os
import random

import numpy as np
from augraphy import *
from augraphy.default.pipeline import *
from PIL import Image
from tqdm import tqdm


def ensure_odd(number):
    """Ensure the number is odd."""
    if number % 2 == 0:
        return number + 1
    return number


def get_ordered_pair(min_val, max_val):
    """Generate a tuple with ordered values."""
    a, b = random.randint(min_val, max_val), random.randint(min_val, max_val)
    return (min(a, b), max(a, b))


def add_noise_to_image(img_path, output_path):
    # Create a new noise model with random parameters for each image
    noise_model = BadPhotoCopy(
        noise_type=random.choice([1, 2, 3, 4, 5]),
        noise_side=random.choice(
            [
                "random",
                "left",
                "top",
                "right",
                "bottom",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
                "none",
                "all",
            ]
        ),
        noise_iteration=get_ordered_pair(2, 5),
        noise_size=get_ordered_pair(2, 6),
        noise_value=get_ordered_pair(random.randint(20, 80), random.randint(150, 200)),
        noise_sparsity=(random.uniform(0.2, 0.6), random.uniform(0.6, 1.0)),
        noise_concentration=(random.uniform(0.2, 0.6), random.uniform(0.6, 1.0)),
        blur_noise=random.choice([-1, 0, 1]),
        blur_noise_kernel=(
            ensure_odd(random.randint(3, 7)),
            ensure_odd(random.randint(3, 7)),
        ),
        wave_pattern=random.choice([-1, 0, 1]),
        edge_effect=random.choice([-1, 0, 1]),
        p=1,  # Ensure the effect is always applied
    )
    img = Image.open(img_path)
    img_np = np.array(img).astype("uint8")
    image_augmented = noise_model(img_np)
    image_binary = image_augmented > 0.5  # Binarization threshold
    image_binary = Image.fromarray(image_binary.astype("uint8") * 255)
    image_binary.save(output_path, format="PNG")


def process_images_in_folder(folder_name):
    folder_path = folder_name
    output_folder = folder_name.replace(
        "clean", "noisy"
    )  # Change clean to noisy for output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            if not os.path.exists(output_path):
                add_noise_to_image(img_path, output_path)


# List of folders to process
folders = ["clean2", "clean3", "clean4"]
for folder in folders:
    process_images_in_folder(folder)
