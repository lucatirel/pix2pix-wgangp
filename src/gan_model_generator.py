import torch.nn as nn
from torch import Tensor


class ResNetBlock(nn.Module):
    """
    A class that represent a single ResNet block, to be used
    in the bottelneck of the autoencoder.
    """

    def __init__(self, in_channels: int):
        """
        Constructs a ResNet block module.

        :param in_channels: The number of input channels.
        """
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)


class ResNet6Generator(nn.Module):
    """
    Constructs a the GAN Generator model as an autoencoder.
    """

    def __init__(self, patch_size: int, use_tanh: bool = False):
        """
        Constructs a ResNet6Generator module.

        :param patch_size: The size of the input patch.
        :param use_tanh: Flag for whether to use Tanh activation in the final layer.
        """
        super(ResNet6Generator, self).__init__()
        self.patch_size = patch_size
        self.use_tanh = use_tanh

        if self.use_tanh is True:
            final_layer = nn.Tanh()
        else:
            final_layer = nn.Sigmoid()

        if self.patch_size == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(128),
            )

            self.residual_blocks = nn.Sequential(
                ResNetBlock(128),
                ResNetBlock(128),
                ResNetBlock(128),
                ResNetBlock(128),
                ResNetBlock(128),
                ResNetBlock(128),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                # nn.Tanh(), QUESTA VIENE USATA NEL PAPER, NOI USIAMO SIGMOIDE
                final_layer,
            )

        elif self.patch_size == 256:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.residual_blocks = nn.Sequential(
                ResNetBlock(512),
                ResNetBlock(512),
                ResNetBlock(512),
                ResNetBlock(512),
                ResNetBlock(512),
                ResNetBlock(512),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                final_layer,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Implements the forward pass of the generator.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


# class ResNet6Generator(nn.Module):
#     """
#     Constructs a the GAN Generator model as an autoencoder.
#     """
#     def __init__(self, patch_size: int):
#         """
#         Constructs a ResNet6Generator module.

#         :param patch_size: The size of the input patch.
#         :param use_tanh: Flag for whether to use Tanh activation in the final layer.
#         """
#         super(ResNet6Generator, self).__init__()
#         self.patch_size = patch_size

#         if self.patch_size == 64:
#             self.encoder = nn.Sequential(
#                 nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(128),
#             )

#             self.residual_blocks = nn.Sequential(
#                 ResNetBlock(128),
#                 ResNetBlock(128),
#                 ResNetBlock(128),
#                 ResNetBlock(128),
#                 ResNetBlock(128),
#                 ResNetBlock(128),
#             )

#             self.decoder = nn.Sequential(
#                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(64),
#                 nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
#                 # nn.Tanh(), QUESTA VIENE USATA NEL PAPER, NOI USIAMO SIGMOIDE
#                 nn.Sigmoid(),
#             )

#         elif self.patch_size == 256:
#             self.encoder = nn.Sequential(
#                 nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(128),
#                 nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(256),
#                 nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(512),
#             )

#             self.residual_blocks = nn.Sequential(
#                 ResNetBlock(512),
#                 ResNetBlock(512),
#                 ResNetBlock(512),
#                 ResNetBlock(512),
#                 ResNetBlock(512),
#                 ResNetBlock(512),
#             )

#             self.decoder = nn.Sequential(
#                 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(256),
#                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(128),
#                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.BatchNorm2d(64),
#                 nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
#                 nn.Sigmoid(),
#             )

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Implements the forward pass of the generator.

#         :param x: Input tensor.
#         :return: Output tensor.
#         """
#         x = self.encoder(x)
#         x = self.residual_blocks(x)
#         x = self.decoder(x)
#         return x
