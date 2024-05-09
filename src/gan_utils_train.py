import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gan_dataloader import ImageDataloader
from gan_loss import GANLoss
from gan_model_discriminator import PatchGANDiscriminator
from gan_model_generator import ResNet6Generator
from gan_utils import create_dirs_tree, save_checkpoint
from pytorch_msssim import ssim
from torch import Tensor
from torch.nn.modules.loss import L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def compute_gradient_penalty(
    discriminator: torch.nn.Module,
    real_samples: Tensor,
    fake_samples: Tensor,
    noisy_samples: Tensor,
    device: torch.device,
) -> Tensor:
    """
    Computes the gradient penalty for a discriminator in a Generative Adversarial Network (GAN).

    Args:
        discriminator (PatchGANDiscriminator): The discriminator model.
        real_samples (torch.Tensor): A batch of real samples.
        fake_samples (torch.Tensor): A batch of fake (generated) samples.
        noisy_samples (torch.Tensor): A batch of noisy samples.
        device (torch.device): The device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: The computed gradient penalty.
    """

    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)

    # Get random interpolation between clean and denoised samples
    interpolates = (
        (alpha * real_samples + ((1 - alpha) * fake_samples))
        .requires_grad_(True)
        .to(device)
    )
    d_interpolates = discriminator(interpolates, noisy_samples)

    fake = (
        torch.Tensor(real_samples.shape[0], 1, 8, 8)
        .fill_(1.0)
        .requires_grad_(False)
        .to(device)
    )

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def initialize_training_pipeline(cfg):
    """
    Initialize the training pipeline for a GAN network.

    Args:
        training_params: A tuple containing the parameters for training. In order:
            batch_size: The number of samples per batch.
            lr_generator: Learning rate for the generator.
            lr_discriminator: Learning rate for the discriminator.
            epochs: The number of epochs for training.
            training_percentage: The percentage of the dataset used for training.
            patch_size: The size of the patches to be extracted from the images.
            crop_size: The size of the random crops to be extracted from the patches for training.
            patch_stride: The stride for the patch extraction.
            lambd: The weight for the L1 loss in the loss function.
            use_wgangp: Whether to use WGAN-GP.
            gp_weight: The gradient penalty weight for WGAN-GP.
            clamp_value: The clamping value for the discriminator's weights for WGAN-GP.
            loss_mode: The type of GAN loss to use.
            use_tanh: Whether to use the tanh activation function in the generator.
        dataset_dir: The directory where the dataset is stored.

    Returns:
        initialization: A list containing all the necessary components for training a GAN.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    generator = ResNet6Generator(
        patch_size=cfg["training"]["patch_size"], use_tanh=cfg["training"]["use_tanh"]
    ).to(device)
    discriminator = PatchGANDiscriminator(patch_size=cfg["training"]["patch_size"]).to(
        device
    )

    (
        clean_folder,
        noise_folder,
        runs_folder,
        current_run_folder,
        tensorboard_logdir,
        checkpoint_dir,
        last_run_folder,
        params_file_path,
    ) = create_dirs_tree(cfg["training"]["dataset_dir"], generator, discriminator, cfg)

    # Load the dataloader
    dataset = ImageDataloader(
        noise_folder,
        clean_folder,
        patch_size=cfg["training"]["patch_size"],
        crop_size=cfg["training"]["crop_size"],
    )

    # Split dataset into training and validation
    train_size = int(cfg["training"]["training_percentage"] * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    # This GANLoss class encapsulates both LSGAN and vanilla GAN loss functions
    criterion_GAN = GANLoss(cfg["training"]["loss_mode"]).to(device)
    criterion_L1 = nn.L1Loss()

    # Separate optimizers for the generator and the discriminator
    optimizer_G = AdamW(
        generator.parameters(), lr=cfg["training"]["lr_generator"], betas=(0.5, 0.999)
    )
    optimizer_D = AdamW(
        discriminator.parameters(),
        lr=cfg["training"]["lr_discriminator"],
        betas=(0.5, 0.999),
    )

    # Defining learning rate schedulers
    # scheduler_G = ReduceLROnPlateau(optimizer_G, "max", patience=5, verbose=True)
    # scheduler_D = ReduceLROnPlateau(optimizer_D, "min", patience=5, verbose=True)
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.1)
    scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.1)
    # scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs-5)
    # scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs-5)

    writer = SummaryWriter(tensorboard_logdir)

    if cfg["training"]["checkpoint_path"]:
        load_full_checkpoint(
            cfg["training"]["checkpoint_path"],
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            device,
        )
    initialization = (
        device,
        generator,
        discriminator,
        train_dataloader,
        valid_dataloader,
        criterion_GAN,
        criterion_L1,
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        writer,
        tensorboard_logdir,
        checkpoint_dir,
    )

    return initialization


def load_full_checkpoint(
    filepath: str,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    scheduler_G: torch.optim.lr_scheduler._LRScheduler,
    scheduler_D: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Tuple[int, float]:
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
        scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_G_loss = checkpoint["best_G_loss_val"]

        print(
            f"Resuming from epoch {start_epoch} with best generator loss of {best_G_loss}"
        )

    else:
        print(f"No checkpoint found at '{filepath}'")


def create_patches(
    img_index: int,
    noisy_imgs: Tensor,
    clean_imgs: Tensor,
    cfg,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Create patches from noisy and clean images using the unfolding method. The resulting patches are moved to the specified device.

    Args:
        img_index (int): The index of the image from which patches are to be created.
        noisy_imgs (Tensor): Tensor of noisy images of shape (N, C, H, W), where
                              N is the batch size, C is the number of channels, H is the height and W is the width.
        clean_imgs (Tensor): Tensor of clean images of the same shape as noisy_imgs.
        patch_size (int): The size of the patches to be created.
        patch_stride (int): The stride to be used when creating patches.
        device (Device): The device to which the patches are to be moved.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor represents patches from noisy images and the second tensor represents patches from clean images. Both tensors have the shape (N*H'*W', 1, patch_size, patch_size), where H' and W' are the new height and width after unfolding.
    """

    patch_size = (cfg["training"]["patch_size"],)
    patch_stride = (cfg["training"]["patch_stride"],)

    noisy_img_patches = (
        noisy_imgs[img_index]
        .unfold(2, patch_size, patch_stride)
        .unfold(3, patch_size, patch_stride)
        .contiguous()
        .view(-1, 1, patch_size, patch_size)
        .to(device)
    )

    clean_img_patches = (
        clean_imgs[img_index]
        .unfold(2, patch_size, patch_stride)
        .unfold(3, patch_size, patch_stride)
        .contiguous()
        .view(-1, 1, patch_size, patch_size)
        .to(device)
    )
    return noisy_img_patches, clean_img_patches


def train_generator(
    generator: ResNet6Generator,
    discriminator: PatchGANDiscriminator,
    optimizer_G: torch.optim.Optimizer,
    adv_loss_criterion: GANLoss,
    l1_loss_criterion: L1Loss,
    noisy_img_patches: Tensor,
    clean_img_patches: Tensor,
    cfg,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Train generator model with respect to the inputs.

    :param generator: The generator model to train.
    :param discriminator: The discriminator model.
    :param optimizer_G: The optimizer for the generator model.
    :param adv_loss_criterion: The criterion for the adversarial loss.
    :param l1_loss_criterion: The criterion for the L1 loss.
    :param noisy_img_patches: The noisy image patches as input to the generator.
    :param clean_img_patches: The clean image patches as target for the L1 loss.
    :param wgangp_flag: A flag indicating whether to use WGAN-GP loss or not.
    :param lambda_value: The weight for L1 loss in the combined loss.
    :return: The combined loss, adversarial loss, L1 loss, and generated image patches.
    """
    wgangp_flag = cfg["training"]["use_wgangp"]
    lambda_value = cfg["training"]["lambd"]

    optimizer_G.zero_grad()

    # IN: BATCHx1x256x256 (REAL_2) -> OUT: BATCHx1x256x256 (REAL_+)
    gen_img_patches = generator(noisy_img_patches)
    # IN: BATCHx2x256x256 (REAL_+) -> OUT: BATCHx1x8x8 (REAL_+-)
    D_fake = discriminator(gen_img_patches, noisy_img_patches)
    if wgangp_flag:
        adv_loss = torch.mean(D_fake, dim=(2, 3))
        adv_loss = -torch.mean(adv_loss)
    else:
        adv_loss = adv_loss_criterion(D_fake, True)

    L1_loss = l1_loss_criterion(gen_img_patches, clean_img_patches)

    G_loss = adv_loss + lambda_value * L1_loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss, adv_loss, L1_loss, gen_img_patches


def train_discriminator(
    discriminator: PatchGANDiscriminator,
    optimizer_D: torch.optim.Optimizer,
    adv_loss_criterion: GANLoss,
    clean_img_patches: Tensor,
    noisy_img_patches: Tensor,
    gen_img_patches: Tensor,
    cfg,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Conducts a single training step for a discriminator model in a Generative Adversarial Network (GAN).

    :param discriminator: Instance of the discriminator model.
    :param optimizer_D: Optimizer used for the discriminator model.
    :param adv_loss_criterion: Criterion to calculate adversarial loss of the discriminator's judgement.
    :param clean_img_patches: Clean image patches used as real examples for the discriminator.
    :param noisy_img_patches: Noisy image patches used along with both real and generated patches.
    :param gen_img_patches: Generated image patches produced by the generator model, used as fake examples for the discriminator.
    :param wgangp_flag: Flag to use Wasserstein GAN with Gradient Penalty (WGAN-GP) as adversarial loss or not.
    :param gp_weight: Weighting factor for the gradient penalty in the total discriminator loss calculation.
    :param clamp_value: The value to which the weights of the discriminator should be clamped in WGAN.
    :param device: The device (CPU or GPU) on which computations should be performed.

    :return: Returns a 4-tuple containing the total discriminator loss, real loss, fake loss, and gradient penalty (if WGAN-GP is used).
    """

    wgangp_flag = cfg["training"]["use_wgangp"]
    gp_weight = cfg["training"]["gp_weight"]
    clamp_value = cfg["training"]["clamp_value"]

    optimizer_D.zero_grad()

    D_real = discriminator(clean_img_patches, noisy_img_patches)
    D_real_loss = adv_loss_criterion(D_real, True)

    D_fake = discriminator(gen_img_patches.detach(), noisy_img_patches)
    D_fake_loss = adv_loss_criterion(D_fake, False)

    if wgangp_flag:
        D_wloss = torch.mean(D_fake, dim=(2, 3)) - torch.mean(D_real, dim=(2, 3))
        D_wloss = torch.mean(D_wloss)

    D_loss = D_wloss if wgangp_flag else (D_real_loss + D_fake_loss) / 2

    if wgangp_flag:
        gradient_penalty = compute_gradient_penalty(
            discriminator,
            clean_img_patches.data,
            gen_img_patches.data,
            noisy_img_patches.data,
            device,
        )
        D_loss += gp_weight * gradient_penalty

    D_loss.backward()
    optimizer_D.step()

    # # Clip weights of discriminator in WGAN
    if wgangp_flag and clamp_value != "":
        for p in discriminator.parameters():
            p.data.clamp_(-clamp_value, clamp_value)

    return D_loss, D_real_loss, D_fake_loss, gradient_penalty if wgangp_flag else None


def update_tensorboard_training_logs(
    i_counter: int,
    epoch_counter: int,
    logging_losses: Tuple[
        float, float, float, float, float, float, float, float, float
    ],
    train_dataloader: DataLoader,
    writer: SummaryWriter,
    wgangp_flag: bool,
) -> None:
    """
    Update the tensorboard logs with the training losses.

    Args:
        i_counter (int): The current iteration (or batch) counter.
        epoch_counter (int): The current epoch counter.
        logging_losses (Tuple[List[float]]): A tuple containing lists of all loss values.
            Each list in the tuple corresponds to a different type of loss.
            The order of lists in the tuple is as follows:
            running_GAN_loss, running_L1_loss, G_loss, GAN_loss, L1_loss,
            D_loss, D_real_loss, D_fake_loss, gradient_penalty.
        train_dataloader (DataLoader): The DataLoader object for the training data.
        writer (SummaryWriter): Tensorboard's writer object.
        wgangp_flag (bool): Flag to determine if Wasserstein GAN with Gradient Penalty (WGAN-GP)
            is used or not. If it's True, the gradient penalty is logged into Tensorboard.

    Returns:
        None
    """

    (
        running_GAN_loss,
        running_L1_loss,
        G_loss,
        GAN_loss,
        L1_loss,
        D_loss,
        D_real_loss,
        D_fake_loss,
        gradient_penalty,
    ) = logging_losses

    avg_GAN_loss = sum(running_GAN_loss) / len(running_GAN_loss)
    avg_L1_loss = sum(running_L1_loss) / len(running_L1_loss)

    print(
        f"[UPDATE] Batch {i_counter+1}/{len(train_dataloader)} GAN Loss (Avg): {avg_GAN_loss:.4f} L1 Loss (Avg): {avg_L1_loss:.4f}"
    )

    # Tensorboard logging
    global_step = epoch_counter * len(train_dataloader) + i_counter
    writer.add_scalar(
        "TRAIN/Loss/Generator_adversarial_avg",
        avg_GAN_loss,
        global_step + 1,
    )
    writer.add_scalar("TRAIN/Loss/Generator_l1_avg", avg_L1_loss, global_step + 1)

    writer.add_scalar("TRAIN/Loss/Generator_total", G_loss, global_step + 1)
    writer.add_scalar("TRAIN/Loss/Generator_adversarial", GAN_loss, global_step + 1)
    writer.add_scalar("TRAIN/Loss/Generator_l1", L1_loss, global_step + 1)

    writer.add_scalar("TRAIN/Loss/Discriminator_total", D_loss, global_step + 1)
    writer.add_scalar("TRAIN/Loss/Discriminator_real", D_real_loss, global_step + 1)
    writer.add_scalar("TRAIN/Loss/Discriminator_fake", D_fake_loss, global_step + 1)
    if wgangp_flag:
        writer.add_scalar(
            "TRAIN/Loss/Discriminator__gradient_penalty",
            gradient_penalty,
            global_step + 1,
        )


def update_tensorboard_imagesgrid(
    generator: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    i_counter: int,
    epoch_counter: int,
    imgs_to_display: int,
    patches_to_display: int,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    global_step = epoch_counter * len(train_dataloader) + i_counter
    with torch.no_grad():
        generator.eval()

        # Select multiple indices here to select the images
        random_indices = torch.randint(
            len(valid_dataloader.dataset),
            size=(10,),
        )

        clean_patches = []
        noised_patches = []
        denoised_patches = []

        # Now iterate over images from validation data
        for idx, img_idx in enumerate(random_indices):
            noised, clean = valid_dataloader.dataset[img_idx.item()]

            noised = noised.to(device).unsqueeze(0)
            clean = clean.to(device).unsqueeze(0)

            # Get random patch to display in tensorboard
            if idx < imgs_to_display:
                patch_indices = torch.randint(0, clean.shape[1], (patches_to_display,))

                for idx in patch_indices:
                    clean_patch = clean[0, idx]
                    noised_patch = noised[0, idx]
                    denoised_patch = generator(noised_patch.unsqueeze(0))

                    if generator.use_tanh:
                        denoised_patch = (denoised_patch > 0).to(denoised_patch.dtype)
                    else:
                        denoised_patch = (denoised_patch > 0.5).to(denoised_patch.dtype)

                    clean_patches.append(clean_patch)
                    noised_patches.append(noised_patch)
                    denoised_patches.append(denoised_patch.squeeze(0))

        # Make a grid from the list of patches
        clean_patches_grid = make_grid(clean_patches, nrow=imgs_to_display)
        noised_patches_grid = make_grid(noised_patches, nrow=imgs_to_display)
        denoised_patches_grid = make_grid(denoised_patches, nrow=imgs_to_display)

        # Concatenate along width axis (dimension -1 for 3D tensor, i.e., channels-height-width)
        concatenated_patches_grid = torch.cat(
            [
                clean_patches_grid,
                denoised_patches_grid,
                noised_patches_grid,
            ],
            -1,
        )

        # Add concatenated image to TensorBoard
        writer.add_image(
            "Images/clean-denoised-noisy",
            concatenated_patches_grid,
            global_step=global_step + 1,
        )

        generator.train()


def save_best_model(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    scheduler_G: torch.optim.lr_scheduler._LRScheduler,
    scheduler_D: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_dataloader: DataLoader,
    G_loss: float,
    avg_ssim: float,
    best_G_loss: float,
    save_dir: str,
    save_counter: int,
    n_save: int,
) -> Tuple[float, int]:
    """
    Saves the best model checkpoint based on the generator loss.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        optimizer_G (torch.optim.Optimizer): The optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): The optimizer for the discriminator.
        epoch (int): The current epoch number.
        val_dataloader (DataLoader): The validation dataloader.
        G_loss (float): The generator loss value for the current epoch.
        avg_ssim (float): The average SSIM (Structural Similarity Index) value for the current epoch.
        best_G_loss (float): The best generator loss value achieved so far.
        save_dir (str): The directory where the model checkpoints will be saved.
        save_counter (int): The counter for the number of saved checkpoints.
        n_save (int): The maximum number of checkpoints to save.

    Returns:
        Tuple[float, int]: A tuple containing the updated best_G_loss value and the incremented save_counter.

    """

    avg_epoch_G_loss_val = G_loss
    if avg_epoch_G_loss_val < best_G_loss:
        best_G_loss = avg_epoch_G_loss_val
        is_best = True
    else:
        is_best = False

    state = {
        "epoch": epoch + 1,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "scheduler_G_state_dict": scheduler_G.state_dict(),
        "scheduler_D_state_dict": scheduler_D.state_dict(),
        "best_G_loss_val": best_G_loss,
    }

    save_checkpoint(
        state, is_best, save_dir, f"checkpoint_{save_counter % n_save}.pth.tar"
    )

    return best_G_loss, save_counter + 1


def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    valid_dataloader: DataLoader,
    criterion_GAN: GANLoss,
    criterion_L1: L1Loss,
    cfg,
    device: torch.device,
) -> Tuple[List[float]]:
    """
    Perform validation on the generator model using the provided validation dataloader.

    Args:
        generator (Module): The generator model to validate.
        discriminator (Module): The discriminator model.
        valid_dataloader (DataLoader): The validation dataloader.
        criterion_GAN (_Loss): The GAN loss function.
        criterion_L1 (_Loss): The L1 loss function.
        lambd (float): The weight parameter for the L1 loss.
        patch_size (int): The size of patches used for training.
        patch_stride (int): The stride for extracting patches.
        device (Device): The device on which to perform the validation.

    Returns:
        Tuple[List[float]]: A tuple of lists containing the average values of various metrics:
            - avg_ssim_val: The average structural similarity index (SSIM) value.
            - avg_GAN_loss_val: The average GAN loss value.
            - avg_L1_loss_val: The average L1 loss value.
            - avg_G_loss_val: The average generator loss value.
            - avg_D_loss_val: The average discriminator loss value.
            - avg_D_real_loss_val: The average discriminator loss on real samples.
            - avg_D_fake_loss_val: The average discriminator loss on fake samples.
    """

    lambd = cfg["training"]["lambd"]
    patch_size = cfg["training"]["patch_size"]
    patch_stride = cfg["training"]["patch_stride"]

    with torch.no_grad():
        generator.eval()

        # Additional logging for validation set metrics
        ssim_values_val = []
        GAN_loss_val = []
        L1_loss_val = []
        G_loss_val = []
        D_loss_val = []
        D_real_loss_val = []
        D_fake_loss_val = []

        for i, (noisy_imgs_val, clean_imgs_val) in enumerate(valid_dataloader):
            batch_size_patches_val = noisy_imgs_val.size(0)

            # For each image in the batch
            for b in range(batch_size_patches_val):
                noisy_img_patches_val = (
                    noisy_imgs_val[b]
                    .unfold(2, patch_size, patch_stride)
                    .unfold(3, patch_size, patch_stride)
                    .contiguous()
                    .view(-1, 1, patch_size, patch_size)
                    .to(device)
                )

                clean_img_patches_val = (
                    clean_imgs_val[b]
                    .unfold(2, patch_size, patch_stride)
                    .unfold(3, patch_size, patch_stride)
                    .contiguous()
                    .view(-1, 1, patch_size, patch_size)
                    .to(device)
                )

                gen_img_patches_val = generator(noisy_img_patches_val)

                val_D_real = discriminator(clean_img_patches_val, noisy_img_patches_val)
                val_D_real_loss = criterion_GAN(val_D_real, True)
                val_D_fake = discriminator(gen_img_patches_val, noisy_img_patches_val)
                val_D_fake_loss = criterion_GAN(val_D_fake, False)
                val_D_loss = (val_D_real_loss + val_D_fake_loss) / 2
                val_GAN_loss = criterion_GAN(val_D_fake, True)
                val_L1_loss = criterion_L1(gen_img_patches_val, clean_img_patches_val)
                val_G_loss = val_GAN_loss + lambd * val_L1_loss

                GAN_loss_val.append(val_GAN_loss.item())
                L1_loss_val.append(val_L1_loss.item())
                G_loss_val.append(val_G_loss.item())
                D_loss_val.append(val_D_loss.item())
                D_real_loss_val.append(val_D_real_loss.item())
                D_fake_loss_val.append(val_D_fake_loss.item())

                val_ssim_value = ssim(
                    clean_img_patches_val, gen_img_patches_val, data_range=1.0
                )
                ssim_values_val.append(val_ssim_value.item())

            if (i + 1) * batch_size_patches_val >= 1024:
                break

        avg_ssim_val = sum(ssim_values_val) / len(ssim_values_val)
        avg_GAN_loss_val = sum(GAN_loss_val) / len(GAN_loss_val)
        avg_L1_loss_val = sum(L1_loss_val) / len(L1_loss_val)
        avg_G_loss_val = sum(G_loss_val) / len(G_loss_val)
        avg_D_loss_val = sum(D_loss_val) / len(D_loss_val)
        avg_D_real_loss_val = sum(D_real_loss_val) / len(D_real_loss_val)
        avg_D_fake_loss_val = sum(D_fake_loss_val) / len(D_fake_loss_val)

        generator.train()

    return (
        avg_ssim_val,
        avg_GAN_loss_val,
        avg_L1_loss_val,
        avg_G_loss_val,
        avg_D_loss_val,
        avg_D_real_loss_val,
        avg_D_fake_loss_val,
    )


def log_validation_results(
    writer: SummaryWriter, epoch: int, validation_results: Tuple[List[float]]
) -> None:
    """
    Log the validation losses and metrics to Tensorboard.

    Args:
        writer (SummaryWriter): The Tensorboard SummaryWriter object.
        epoch (int): The current epoch number.
        validation_results (Tuple[List[float]]): A tuple of lists containing the average values of various metrics:
            - avg_ssim_val: The average structural similarity index (SSIM) value.
            - avg_GAN_loss_val: The average GAN loss value.
            - avg_L1_loss_val: The average L1 loss value.
            - avg_G_loss_val: The average generator loss value.
            - avg_D_loss_val: The average discriminator loss value.
            - avg_D_real_loss_val: The average discriminator loss on real samples.
            - avg_D_fake_loss_val: The average discriminator loss on fake samples.
    """

    (
        avg_ssim_val,
        avg_GAN_loss_val,
        avg_L1_loss_val,
        avg_G_loss_val,
        avg_D_loss_val,
        avg_D_real_loss_val,
        avg_D_fake_loss_val,
    ) = validation_results

    # Logging validation losses and metrics to Tensorboard
    writer.add_scalar("VAL/SSIM_avg", avg_ssim_val, epoch)
    writer.add_scalar("VAL/Loss/GAN_loss_val", avg_GAN_loss_val, epoch)
    writer.add_scalar("VAL/Loss/L1_loss_val", avg_L1_loss_val, epoch)
    writer.add_scalar("VAL/Loss/G_loss_val", avg_G_loss_val, epoch)
    writer.add_scalar("VAL/Loss/D_loss_val", avg_D_loss_val, epoch)
    writer.add_scalar("VAL/Loss/D_real_loss_val", avg_D_real_loss_val, epoch)
    writer.add_scalar("VAL/Loss/D_fake_loss_val", avg_D_fake_loss_val, epoch)
