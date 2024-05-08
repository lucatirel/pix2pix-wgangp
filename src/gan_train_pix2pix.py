import os

import torch
import torch.nn as nn
from alive_progress import alive_bar
from gan_loss import GANLoss
from gan_utils_train import (
    create_patches,
    initialize_training_pipeline,
    log_validation_results,
    save_best_model,
    train_discriminator,
    train_generator,
    update_tensorboard_imagesgrid,
    update_tensorboard_training_logs,
    validate,
)
from torch.nn.modules.loss import L1Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CuDNN Version:", torch.backends.cudnn.version())
print("CUDA Version:", torch.version.cuda)


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    criterion_GAN: GANLoss,
    criterion_L1: L1Loss,
    optimizer_G: Optimizer,
    optimizer_D: Optimizer,
    scheduler_G: _LRScheduler,
    scheduler_D: _LRScheduler,
    writer: SummaryWriter,
    device: torch.device,
    epochs: int = 20,
    tensorboard_logdir: str = "logs",
    patch_size: int = 64,
    patch_stride: int = 32,
    save_dir: str = "./",
    n_save: int = 5,
    lambd: float = 100,
    imgs_to_display: int = 1,
    patches_to_display: int = 5,
    use_wgangp: bool = False,
    gp_weight: float = 10,
    clamp_value: float = 0.01,
    training_log_step: int = 20,
) -> None:
    best_G_loss = float("inf")
    save_counter = 0

    for epoch in range(epochs):
        print("\n" + "=" * 15 + f" Epoch {epoch+1}/{epochs} " + "=" * 15)
        running_GAN_loss = []
        running_L1_loss = []
        epoch_G_loss = 0

        # For each (noisy, clean) images batch in epoch
        total_iterations = len(train_dataloader)
        with alive_bar(total_iterations, force_tty=True) as bar:
            for i, (noisy_imgs, clean_imgs) in enumerate(train_dataloader):

                try:
                    batch_size_patches = noisy_imgs.size(0)

                    # For each image in the batch
                    for img_index in range(batch_size_patches):
                        print(f"Batch Step {img_index}/{batch_size_patches}")
                        noisy_img_patches, clean_img_patches = create_patches(
                            img_index,
                            noisy_imgs,
                            clean_imgs,
                            patch_size,
                            patch_stride,
                            device,
                        )

                        # TRAIN GENERATOR
                        G_loss, GAN_loss, L1_loss, gen_img_patches = train_generator(
                            generator,
                            discriminator,
                            optimizer_G,
                            criterion_GAN,
                            criterion_L1,
                            noisy_img_patches,
                            clean_img_patches,
                            use_wgangp,
                            lambd,
                        )

                        # TRAIN DISCRIMINATOR
                        (
                            D_loss,
                            D_real_loss,
                            D_fake_loss,
                            gradient_penalty,
                        ) = train_discriminator(
                            discriminator,
                            optimizer_D,
                            criterion_GAN,
                            clean_img_patches,
                            noisy_img_patches,
                            gen_img_patches,
                            use_wgangp,
                            gp_weight,
                            clamp_value,
                            device,
                        )

                        running_GAN_loss.append(GAN_loss.item())
                        running_L1_loss.append(L1_loss.item())

                    # print(
                    #     f"Batch {i+1}/{len(train_dataloader)} Loss_D: {D_loss:.4f} Loss_G: {G_loss:.4f}"
                    # )
                    epoch_G_loss += G_loss

                    # TRAINING LOGS + INFERENCE STEP
                    if (i + 1) % training_log_step == 0:
                        logging_losses = (
                            running_GAN_loss,
                            running_L1_loss,
                            G_loss,
                            GAN_loss,
                            L1_loss,
                            D_loss,
                            D_real_loss,
                            D_fake_loss,
                            gradient_penalty,
                        )

                        update_tensorboard_training_logs(
                            i,
                            epoch,
                            logging_losses,
                            train_dataloader,
                            writer,
                            use_wgangp,
                        )

                        running_GAN_loss, running_L1_loss = [], []

                        # INFERENCE STEP ON VALIDATION IMAGES
                        update_tensorboard_imagesgrid(
                            generator,
                            train_dataloader,
                            valid_dataloader,
                            i,
                            epoch,
                            imgs_to_display,
                            patches_to_display,
                            writer,
                            device,
                        )

                    bar()

                except Exception as e:
                    print(
                        f"!!!!!!!!!!!!!!! \nERROR IN THE BATCH {i}: {str(e)} \n!!!!!!!!!!!!!!!"
                    )
                    raise

        # Step the schedulers
        scheduler_G.step()
        scheduler_D.step()

        # VALIDATION STEP
        validation_results = validate(
            generator,
            discriminator,
            valid_dataloader,
            criterion_GAN,
            criterion_L1,
            lambd,
            patch_size,
            patch_stride,
            device,
        )

        avg_ssim_val, avg_G_loss_val = validation_results[0], validation_results[3]

        # Save best generator and discriminator models
        best_G_loss, save_counter = save_best_model(
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            epoch,
            valid_dataloader,
            avg_G_loss_val,
            avg_ssim_val,
            best_G_loss,
            save_dir,
            save_counter,
            n_save,
        )

        save_counter += 1

        # Log validation results
        log_validation_results(writer, epoch, validation_results)

    writer.flush()
    writer.close()


# =====================================================================================

# PATHS COMPUTER LUCA
dataset_dir = os.path.join(
    os.getcwd(), "dataset"
)  # R"C:\Users\Luca Tirel\Desktop\Denoising\datasets"

# PARAMETRI TRAINING
batch_size = 64  # 32  # 128
lr_generator = 0.00002  # 0.0002
lr_discriminator = 0.000002  # 0.0002
epochs = 100
training_percentage = 0.8
patch_size = 256
crop_size = 1024
patch_stride = 128
lambd = 10  # 30_000
use_wgangp = True
gp_weight = 100
clamp_value = 0.01
loss_mode = "vanilla"
training_log_step = 5
use_tanh = False
checkpoint_path = R""

training_params = (
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
    checkpoint_path,
)


if __name__ == "__main__":
    (
        device,
        generator,
        discriminator,
        dataset,
        train_dataloader,
        valid_dataloader,
        criterion_GAN,
        criterion_L1,
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        writer,
        clean_folder,
        noise_folder,
        runs_folder,
        current_run_folder,
        tensorboard_logdir,
        checkpoint_dir,
        last_run_folder,
        params_file_path,
    ) = initialize_training_pipeline(training_params, dataset_dir)

    train(
        generator=generator,
        discriminator=discriminator,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion_GAN=criterion_GAN,
        criterion_L1=criterion_L1,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        writer=writer,
        device=device,
        epochs=epochs,
        tensorboard_logdir=tensorboard_logdir,
        save_dir=checkpoint_dir,
        lambd=lambd,
        patch_size=patch_size,
        patch_stride=patch_stride,
        use_wgangp=use_wgangp,
        n_save=epochs,
        gp_weight=gp_weight,
        clamp_value=clamp_value,
        training_log_step=training_log_step,
    )
