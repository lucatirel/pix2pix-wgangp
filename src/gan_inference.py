import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from gan_utils import *  # noqa: F403
from skimage import metrics
from skimage.util import view_as_windows


def merge_patches(patches, rows, cols, patch_size, stride, apply_window=False):
    # Inizializza un'immagine vuota e un conteggio di sovrapposizioni
    output_image = np.zeros((rows, cols))
    overlap_count = np.zeros((rows, cols))

    # Crea una finestra di Hann della stessa dimensione del patch
    window = np.hanning(patch_size)
    window_2d = np.outer(window, window)

    # Numero di patch per riga/colonna
    patches_per_row = (rows - patch_size) // stride + 1
    patches_per_col = (cols - patch_size) // stride + 1

    patch_idx = 0

    # Posiziona ciascuna patch nella posizione corrispondente nell'immagine output
    for i in range(patches_per_row):
        for j in range(patches_per_col):
            patch_start_i = i * stride
            patch_start_j = j * stride

            patch = patches[patch_idx]
            if apply_window:
                patch = patch * window_2d  # Applica la finestra alla patch
            patch_idx += 1

            output_image[
                patch_start_i : patch_start_i + patch_size,
                patch_start_j : patch_start_j + patch_size,
            ] += patch
            if apply_window:
                overlap_count[
                    patch_start_i : patch_start_i + patch_size,
                    patch_start_j : patch_start_j + patch_size,
                ] += window_2d  # Conta le sovrapposizioni con la finestra
            else:
                overlap_count[
                    patch_start_i : patch_start_i + patch_size,
                    patch_start_j : patch_start_j + patch_size,
                ] += 1

    # Divide ogni pixel per il conteggio di sovrapposizioni per ottenere la media, se è zero viene impostato a uno
    output_image /= np.where(overlap_count != 0, overlap_count, 1)

    return output_image


def main_inference(
    checkpoint_path,
    imgs_path,
    patch_size=256,
    stride=128,
    with_true_clean=True,
    denoising_steps=1,
    apply_smooth_window=False,
    use_tanh=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_models_from_checkpoint(checkpoint_path, device, patch_size)
    generator.eval()

    noisy_imgs_path = os.path.join(imgs_path, "noisy")
    denoised_imgs_path = os.path.join(imgs_path, "denoised")
    os.makedirs(denoised_imgs_path, exist_ok=True)

    if with_true_clean:
        clean_imgs_path = os.path.join(imgs_path, "clean")
        ssim_log_path = os.path.join(imgs_path, "ssim_log.csv")
        with open(ssim_log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "SSIM"])

    total_images = len(os.listdir(noisy_imgs_path))
    for index, filename in enumerate(os.listdir(noisy_imgs_path)):
        denoised_file_path = os.path.join(denoised_imgs_path, filename)
        if not os.path.exists(denoised_file_path):
            noisy_img = cv2.imread(
                os.path.join(noisy_imgs_path, filename), cv2.IMREAD_UNCHANGED
            )
            noisy_img[noisy_img == 255] = 1

            # Find the real clean image
            if with_true_clean:
                clean_img = cv2.imread(
                    os.path.join(clean_imgs_path, filename), cv2.IMREAD_UNCHANGED
                )

            # Calcola l'amount di padding necessario
            rows_extra = patch_size - (noisy_img.shape[0] % patch_size)
            cols_extra = patch_size - (noisy_img.shape[1] % patch_size)

            padded_img = np.pad(
                noisy_img,
                ((0, rows_extra), (0, cols_extra)),
                mode="constant",
                constant_values=0,
            )

            # Divide in patch di dimensione 64x64 con sovrapposizione
            patches = view_as_windows(padded_img, (patch_size, patch_size), step=stride)

            total_patches = patches.shape[0] * patches.shape[1]

            for ds in range(denoising_steps):
                denoised_patches = []
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j]
                        # Resize the patch from (64,64) to (1,64,64) to be able to pass it to the model
                        patch = np.expand_dims(patch, axis=0)
                        patch = np.expand_dims(patch, axis=0)
                        print(
                            f"Denoising patch {i*patches.shape[1]+j+1}/{total_patches} at step {ds+1}/{denoising_steps} of image {index+1}/{total_images}"
                        )
                        denoised_patch = generator(torch.from_numpy(patch).float())
                        denoised_patch = denoised_patch.squeeze().cpu().detach().numpy()
                        if generator.use_tanh:
                            denoised_patch = np.where(denoised_patch > 0, 1, 0).astype(
                                np.uint8
                            )
                        else:
                            denoised_patch = np.where(
                                denoised_patch > 0.5, 1, 0
                            ).astype(np.uint8)
                        denoised_patches.append(denoised_patch)

            # Riunisce le patch denoised per formare l'immagine originale
            denoised_image = merge_patches(
                denoised_patches,
                noisy_img.shape[0] + rows_extra,
                noisy_img.shape[1] + cols_extra,
                patch_size,
                stride,
                apply_window=apply_smooth_window,
            )

            # Rimuove il padding aggiunto precedentemente
            denoised_image = denoised_image[: noisy_img.shape[0], : noisy_img.shape[1]]
            denoised_image = (
                np.where(denoised_image >= 0.5, 1, 0).astype(np.uint8) * 255
            )

            # After generating the denoised image:
            cv2.imwrite(denoised_file_path, denoised_image)

            if with_true_clean:
                # Calculate SSIM
                ssim_value = metrics.structural_similarity(
                    clean_img,
                    denoised_image,
                    data_range=denoised_image.max() - denoised_image.min(),
                )

                # Log SSIM to CSV
                with open(ssim_log_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, ssim_value])

            # Display the original and denoised images
            # if with_true_clean:
            #     plt.figure(figsize=(15, 5))

            #     plt.subplot(1, 3, 1)
            #     plt.title("Clean Image (True)")
            #     plt.imshow(clean_img, cmap="gray")

            #     plt.subplot(1, 3, 2)
            #     plt.title("Clean Image (Denoised)")
            #     plt.imshow(denoised_image, cmap="gray")

            #     plt.subplot(1, 3, 3)
            #     plt.title("Noisy Image (True)")
            #     plt.imshow(noisy_img, cmap="gray")

            # else:
            #     plt.figure(figsize=(10, 5))

            #     plt.subplot(1, 2, 1)
            #     plt.title("Clean Image (Denoised)")
            #     plt.imshow(denoised_image, cmap="gray")

            #     plt.subplot(1, 2, 2)
            #     plt.title("Noisy Image (True)")
            #     plt.imshow(noisy_img, cmap="gray")

            # plt.show()


checkpoint_path = R"C:\Users\Luca\Desktop\pix2pix-wgangp\runs\run_denoisegan_20240507-234550\checkpoints\model_best.pth.tar"
testing_imgs_path = R"C:\Users\Luca\Desktop\pix2pix-wgangp\dataset\testing"

CLEANING_STRIDE = 128
DENOISING_STEPS = 1
APPLY_SMOOTH_WINDOW = False
USE_TANH = False

if __name__ == "__main__":
    try:
        main_inference(
            checkpoint_path,
            testing_imgs_path,
            stride=CLEANING_STRIDE,
            with_true_clean=True,
            denoising_steps=DENOISING_STEPS,
            apply_smooth_window=APPLY_SMOOTH_WINDOW,
            use_tanh=USE_TANH,
        )
    except Exception as exc:
        print(exc)
        breakpoint()
