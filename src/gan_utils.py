import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torchsummary import summary

from gan_model_discriminator import PatchGANDiscriminator
from gan_model_generator import ResNet6Generator
from gan_model_wrapper import WrappedModel


def clean_directory(directory_path: str) -> None:
    """
    Deletes all files and directories in the specified directory.

    Args:
        directory_path (str): Path to the directory to be cleaned.

    Raises:
        Exception: If any error occurs during deletion.
    """
    os.makedirs(directory_path, exist_ok=True)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_path: str = "./",
    filename: str = "checkpoint.pth.tar",
) -> None:
    """
    Saves the state in a file. If the state is the best one, it saves the state in another file as well.

    Args:
        state (StateDict): The state to be saved.
        is_best (bool): Whether the current state is the best or not.
        checkpoint_path (str, optional): Path to the directory where the state will be saved. Defaults to "./".
        filename (str, optional): Name of the file where the state will be saved. Defaults to "checkpoint.pth.tar".
    """
    saving_path = os.path.join(checkpoint_path, filename)
    torch.save(state, saving_path)
    if is_best:
        saving_path = os.path.join(checkpoint_path, "model_best.pth.tar")
        torch.save(state, saving_path)


def load_models_from_checkpoint(
    filepath: str, device: torch.device, patch_size: int = 256
) -> Union[ResNet6Generator, None]:
    """
    Loads the generator model from the given checkpoint file.

    Args:
        filepath (str): Path to the checkpoint file.
        device (Union[str, torch.device]): Device where the tensors will be allocated.
        patch_size (int, optional): Size of the patch for the ResNet6Generator. Defaults to 256.

    Returns:
        Union[torch.nn.Module, None]: Returns the loaded generator model if the checkpoint file exists, otherwise None.
    """
    generator = ResNet6Generator(patch_size)

    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        start_epoch = checkpoint["epoch"]

        generator.load_state_dict(checkpoint["generator_state_dict"])

        best_G_loss = checkpoint["best_G_loss"]

        print(
            f"Loaded checkpoint '{filepath}', starting from epoch {start_epoch} with best generator loss {best_G_loss}"
        )

        return generator
    else:
        print(f"No checkpoint found at '{filepath}'")
        return None


def create_dirs_tree(
    dataset_dir: str,
    generator: ResNet6Generator,
    discriminator: PatchGANDiscriminator,
    params: Tuple[
        int, float, float, int, float, int, int, int, int, bool, int, float, str, bool
    ],
) -> List[str]:
    """
    Function to create the directories for the clean and noisy images, runs, logs, checkpoints, and parameters file.

    Args:
    dataset_dir (str): Path of the directory with the dataset.
    generator (Module): The generator model of the GAN.
    discriminator (Module): The discriminator model of the GAN.
    params (Tuple): Tuple containing hyperparameters of the model in the order:
        - batch_size (int)
        - lr_generator (float)
        - lr_discriminator (float)
        - epochs (int)
        - training_percentage (float)
        - patch_size (int)
        - crop_size (int)
        - patch_stride (int)
        - lambd (int)
        - use_wgangp (bool)
        - gp_weight (int)
        - clamp_value (float)
        - loss_mode (str)
        - use_tanh (bool)

    Returns:
    dirs_paths (List[str]): List of created directories' paths.
    """
    (
        batch_size,
        lr_generator,
        lr_discriminator,
        epochs,
        training_percentage,
        patch_size,
        crop_size,
        patch_stride,
        lambd,
        use_wgangp,
        gp_weight,
        clamp_value,
        loss_mode,
        use_tanh,
    ) = params

    model_D = WrappedModel(discriminator)

    # get the folders paths of the clean and noisy dirs
    clean_folder = os.path.join(dataset_dir, "clean")
    noise_folder = os.path.join(dataset_dir, "noisy")

    # get the directory where script is located
    base_dir = os.getcwd()

    # get subdirectories paths
    runs_folder = os.path.join(base_dir, "runs")
    os.makedirs(runs_folder, exist_ok=True)

    # define current run directories and create them if they not exists
    current_run_folder = os.path.join(
        runs_folder, f'run_denoisegan_{time.strftime("%Y%m%d-%H%M%S")}'
    )
    tensorboard_logdir = os.path.join(current_run_folder, "logs")
    checkpoint_dir = os.path.join(current_run_folder, "checkpoints")
    os.makedirs(current_run_folder, exist_ok=True)
    os.makedirs(tensorboard_logdir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # clean last run folder and link it to current run
    last_run_folder = os.path.join(base_dir, "last_run")
    if os.path.islink(last_run_folder):
        os.unlink(last_run_folder)
    os.symlink(tensorboard_logdir, last_run_folder)

    # save the current parameters
    params_file_path = os.path.join(current_run_folder, "params.txt")
    with open(params_file_path, "w", encoding="utf-8") as f:
        f.write(f"epochs: {epochs}\n")
        f.write(f"training_percentage: {training_percentage}\n")

        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr_generator: {lr_generator}\n")
        f.write(f"lr_discriminator: {lr_discriminator}\n")
        f.write(f"lambd: {lambd}\n")
        f.write(f"use_wgangp: {use_wgangp}\n")
        f.write(f"gp_weight: {gp_weight}\n")
        f.write(f"clamp_value: {clamp_value}\n")
        f.write(f"loss_mode: {loss_mode}\n")

        f.write(f"patch_size: {patch_size}\n")
        f.write(f"crop_size: {crop_size}\n")
        f.write(f"patch_stride: {patch_stride}\n")
        f.write(f"use_tanh: {use_tanh}\n")

        original_stdout = sys.stdout
        sys.stdout = f

        f.write("\nSummary of Generator:\n")
        summary(generator, (1, patch_size, patch_size))
        f.write("\nSummary of Discriminator:\n")
        summary(model_D, (2, patch_size, patch_size))

        # Reset the standard output to its original value
        sys.stdout = original_stdout

    dirs_paths = [
        clean_folder,
        noise_folder,
        runs_folder,
        current_run_folder,
        tensorboard_logdir,
        checkpoint_dir,
        last_run_folder,
        params_file_path,
    ]
    return dirs_paths
