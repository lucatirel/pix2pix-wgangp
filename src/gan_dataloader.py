import os
import random
from typing import Tuple

import cv2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class ImageDataloader(Dataset):
    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        transform: bool = True,
        patch_size: int = 64,
        patched: bool = True,
        crop_size: int = 512,
        perform_flip_rot_transform: bool = True,
    ):
        """
        Initializes the dataset loader with the given parameters.

        Parameters:
        - noisy_dir: Path to the directory of noisy images.
        - clean_dir: Path to the directory of clean images.
        - transform: Whether or not to apply transforms on images.
        - patch_size: The size of the patches to be extracted from each image.
        - patched: Whether or not to split the image into patches.
        - crop_size: The size of the cropped image if transform is True.
        - perform_flip_rot_transform: Whether or not to perform flip and rotat. transf.
        """

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.perform_flip_rot_transform = perform_flip_rot_transform
        self.patch_size = patch_size
        self.patched = patched
        self.crop_size = crop_size

        self.noisy_ids = os.listdir(noisy_dir)
        self.clean_ids = os.listdir(clean_dir)

        self.to_tensor_transform = transforms.ToTensor()

    def __len__(self) -> int:
        """Returns the length of the dataset, ie, number of noisy (or clean) images."""
        return len(self.noisy_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves a noisy-clean image pair based on the given index.

        The method returns a tuple of two torch tensors, where each tensor
        is a collection of image patches. Each patch is a 3D tensor, where the
        first dimension is the number of channels (1 for grayscale images),
        and the remaining dimensions are the height and width of the patch.

        If the image cannot be cropped, the method tries to retrieve the next image.

        Parameters:
        - index: Index of the noisy-clean image pair to be retrieved.

        Returns:
        - A tuple of two torch tensors (noisy_patches, clean_patches). Each tensor
        is a collection of image patches, where each patch is a 3D tensor (number
        of channels, height, width).

        Raises:
        - RuntimeError: If the noisy or clean image cannot be read.
        """

        while True:
            noisy_img_path = os.path.join(self.noisy_dir, self.noisy_ids[index])
            clean_img_path = os.path.join(self.clean_dir, self.clean_ids[index])

            try:
                noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_UNCHANGED)
                if noisy_img is None:
                    raise RuntimeError(f"Error reading noisy image: {noisy_img_path}")

                clean_img = cv2.imread(clean_img_path, cv2.IMREAD_UNCHANGED)
                if clean_img is None:
                    raise RuntimeError(f"Error reading clean image: {clean_img_path}")

                if self.transform:
                    # Convert images to PIL
                    noisy_img = transforms.ToPILImage()(noisy_img)
                    clean_img = transforms.ToPILImage()(clean_img)

                    # Apply the SAME random crop to both images
                    i, j, h, w = transforms.RandomCrop.get_params(
                        noisy_img, output_size=(self.crop_size, self.crop_size)
                    )
                    noisy_img = F.crop(noisy_img, i, j, h, w)
                    clean_img = F.crop(clean_img, i, j, h, w)

                    # Convert images back to tensors
                    noisy_img = self.to_tensor_transform(noisy_img)
                    clean_img = self.to_tensor_transform(clean_img)

                if self.patched:
                    noisy_patches = self.split_into_patches(noisy_img)
                    clean_patches = self.split_into_patches(clean_img)

                    if self.perform_flip_rot_transform:
                        for i in range(noisy_patches.shape[0]):
                            # Convert patches back to PIL Image for transformations
                            noisy_patch = transforms.ToPILImage()(noisy_patches[i])
                            clean_patch = transforms.ToPILImage()(clean_patches[i])

                            # Apply same flipping and rotation to both patches
                            angle = random.randint(0, 3) * 90
                            noisy_patch = F.rotate(noisy_patch, angle)
                            clean_patch = F.rotate(clean_patch, angle)

                            # Apply same horizontal and vertical flipping
                            if random.random() > 0.75:
                                noisy_patch = F.hflip(noisy_patch)
                                clean_patch = F.hflip(clean_patch)

                            if random.random() > 0.75:
                                noisy_patch = F.vflip(noisy_patch)
                                clean_patch = F.vflip(clean_patch)

                            # Convert patches back to tensors
                            noisy_patches[i] = self.to_tensor_transform(noisy_patch)
                            clean_patches[i] = self.to_tensor_transform(clean_patch)

                    return noisy_patches, clean_patches

                else:
                    return noisy_img, clean_img

            except Exception:
                # DON'T REMOVE
                # print(
                #     f"Error loading images at index {index}: {str(e)}. Skip to next."
                # )
                index = (index + 1) % len(self)

    def split_into_patches(self, img: Tensor) -> Tensor:
        """
        Splits a single image into multiple patches.

        The method takes a 3D tensor representing a grayscale image
        and returns a 4D tensor representing multiple image patches.
        Each patch is a 3D tensor, where the first dimension is the number of channels
        (1 for grayscale images), and the remaining dimensions are the patch height
        and width.

        Parameters:
        - img: A 3D torch tensor representing a grayscale image. The first dimension
        is the number of channels (1 for grayscale images), and the remaining dimensions
        are the image height and width.

        Returns:
        - A 4D torch tensor representing multiple image patches. The first dimension
        is the number of patches, the second dimension is the number of channels
        (1 for grayscale patches), and the remaining dimensions are the patch height
        and width.
        """

        h, w = img.shape[-2:]

        # Calculate the number of patches
        n_patches = (h // self.patch_size) * (w // self.patch_size)

        # Reshape image into patches
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.permute(1, 2, 0, 3, 4).reshape(
            n_patches, 1, self.patch_size, self.patch_size
        )

        return patches
