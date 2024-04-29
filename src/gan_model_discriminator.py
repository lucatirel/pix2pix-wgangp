import torch
import torch.nn as nn
from torch import Tensor


class PatchGANDiscriminator(nn.Module):
    """
    PatchGANDiscriminator Class:
    A class to build the discriminator in PatchGAN.

    Attributes:
    patch_size (int): The size of the patches.
    main (nn.Module): The main neural network module.
    """

    def __init__(self, patch_size: int):
        """
        PatchGANDiscriminator constructor

        Parameters:
        patch_size (int): The size of the patches.
        """
        super(PatchGANDiscriminator, self).__init__()

        self.patch_size = patch_size

        if self.patch_size == 64:
            self.main = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                # nn.Sigmoid(),
            )

        elif self.patch_size == 256:
            self.main = nn.Sequential(
                nn.Conv2d(
                    2, 64, kernel_size=4, stride=2, padding=1
                ),  # output: 16x64x128x128
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    64, 128, kernel_size=4, stride=2, padding=1
                ),  # output: 16x128x64x64
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    128, 256, kernel_size=4, stride=2, padding=1
                ),  # output: 16x256x32x32
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256, 512, kernel_size=4, stride=2, padding=1
                ),  # output: 16x512x16x16
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    512, 512, kernel_size=4, stride=2, padding=1
                ),  # output: 16x512x8x8
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    512, 1, kernel_size=3, stride=1, padding=1
                ),  # output: 16x1x8x8
                # nn.Sigmoid(),
            )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward method: Concatenates the input and target tensors along the channel dimension
        and passes them through the main neural network module.

        Parameters:
        input (Tensor): The input tensor.
        target (Tensor): The target tensor.

        Returns:
        Tensor: The result tensor.
        """
        x = torch.cat([input, target], 1)
        x = self.main(x)
        return x


# class PatchGANDiscriminator(nn.Module):
#     """
#     PatchGANDiscriminator Class:
#     A class to build the discriminator in PatchGAN.

#     Attributes:
#     patch_size (int): The size of the patches.
#     main (nn.Module): The main neural network module.
#     """

#     def __init__(self, patch_size: int):
#         """
#         PatchGANDiscriminator constructor

#         Parameters:
#         patch_size (int): The size of the patches.
#         """
#         super(PatchGANDiscriminator, self).__init__()

#         self.patch_size = patch_size

#         if self.patch_size == 64:
#             self.main = nn.Sequential(
#                 nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
#                 # nn.Sigmoid(),
#             )

#         elif self.patch_size == 256:
#             self.main = nn.Sequential(
#                 nn.Conv2d(
#                     2, 64, kernel_size=4, stride=2, padding=1
#                 ),  # output: 16x64x128x128
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     64, 128, kernel_size=4, stride=2, padding=1
#                 ),  # output: 16x128x64x64
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     128, 256, kernel_size=4, stride=2, padding=1
#                 ),  # output: 16x256x32x32
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     256, 512, kernel_size=4, stride=2, padding=1
#                 ),  # output: 16x512x16x16
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     512, 512, kernel_size=4, stride=2, padding=1
#                 ),  # output: 16x512x8x8
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     512, 1, kernel_size=3, stride=1, padding=1
#                 ),  # output: 16x1x8x8
#                 # nn.Sigmoid(),
#             )

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         """
#         Forward method: Concatenates the input and target tensors along the channel dimension
#         and passes them through the main neural network module.

#         Parameters:
#         input (Tensor): The input tensor.
#         target (Tensor): The target tensor.

#         Returns:
#         Tensor: The result tensor.
#         """
#         x = torch.cat([input, target], 1)
#         x = self.main(x)
#         return x
