import torch.nn as nn
from torch import Tensor

from gan_model_discriminator import PatchGANDiscriminator


class WrappedModel(nn.Module):
    """
    A class used to wrap a PyTorch module and modify its forward method.

    This class is a subclass of PyTorch's nn.Module, and its purpose is to take
    an input tensor, split it into two halves along the second dimension, and
    pass these halves as separate arguments to the wrapped module's forward method.

    ...

    Attributes
    ----------
    module : nn.Module
        the PyTorch module to wrap and whose forward method to modify

    Methods
    -------
    forward(x: Tensor) -> Tensor:
        Overrides the forward method of nn.Module. Splits the input tensor into two halves
        along the second dimension and passes these halves to the wrapped module's forward
        method.
    """

    def __init__(self, module: PatchGANDiscriminator):
        """
        Parameters
        ----------
        module : nn.Module
            the PyTorch module to wrap and whose forward method to modify
        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        """
        Overrides the forward method of nn.Module.

        Takes an input tensor, splits it into two halves along the second dimension, and
        passes these halves as separate arguments to the wrapped module's forward method.

        Parameters
        ----------
        x : Tensor
            the input tensor

        Returns
        ----------
        Tensor
            the output of the wrapped module's forward method
        """
        # split the input tensor into two
        n = int(x.shape[1] / 2)
        input = x[:, :n, :, :]
        target = x[:, n:, :, :]
        return self.module(input, target)
