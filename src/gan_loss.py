import torch
import torch.nn as nn
from torch import Tensor


class GANLoss(nn.Module):
    """
    This class represents a GAN loss module.

    Attributes
    ----------
    real_label : Tensor
        The label representing real images.
    fake_label : Tensor
        The label representing fake images.
    loss : torch.nn.Module
        The loss function. This could be Mean Squared Error Loss (nn.MSELoss) for 'lsgan' mode or
        Binary Cross Entropy with Logits Loss (nn.BCEWithLogitsLoss) for 'vanilla' mode.
    """

    def __init__(self, gan_mode: str, real_label: float = 1.0, fake_label: float = 0.0):
        """
        Initialize the GANLoss module.

        Parameters
        ----------
        gan_mode : str
            The mode of the GAN, which could be 'lsgan' or 'vanilla'.
        real_label : float, optional
            The label representing real images, by default 1.0.
        fake_label : float, optional
            The label representing fake images, by default 0.0.
        """

        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        """
        Get the target tensor.

        Parameters
        ----------
        prediction : Tensor
            The prediction tensor.
        target_is_real : bool
            A boolean indicating if the target is real.

        Returns
        -------
        Tensor
            The target tensor.
        """

        if target_is_real:
            target_tensor = self.real_label.repeat(prediction.size())
        else:
            target_tensor = self.fake_label.repeat(prediction.size())

        return target_tensor

    def __call__(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        """
        Calculate the loss.

        Parameters
        ----------
        prediction : torch.Tensor
            The prediction tensor.
        target_is_real : bool
            A boolean indicating if the target is real.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """

        # IN: BATCHx1x8x8 (REAL_+-) -> OUT: BATCHx1 (REAL_+-)
        prediction_tensor = torch.mean(prediction, dim=(2, 3))
        target_tensor = self.get_target_tensor(prediction_tensor, target_is_real)

        # IN: BATCHx1 (REAL_+-), BATCHx1 (REAL_2) -> OUT: 1 (REAL_2)
        loss = self.loss(prediction_tensor, target_tensor)

        return loss
