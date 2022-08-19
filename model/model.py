import time
from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from encoders import VisionTransformer, JoyStickEncoder
from data import create_dataloader


class CLIPLoss(nn.Module):
    """ Loss function for model, from OpenAI's CLIP paper
    """

    def __init__(self, temperature=0.07) -> None:
        """ Initialize a pytorch lightning module to learn a logit scale

        Args:
            temperature (float, optional): parameter from OpenAI clip paper.
                Used to scale the logits. Defaults to 0.7.
        """
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / temperature))

    def forward(self, lidar_features, joystick_features) -> float:
        """ calculate the loss given a set of lidar feature vectors and joystick feature vectors

        Args:
            lidar_features (_type_): output of lidar_encoder (batch_size, output_dim)
            joystick_features (_type_): output of joystick encoder (batch_size, output_dim)

        Returns:
            float: total loss
        """
        assert lidar_features.shape == joystick_features.shape

        # L2 normalization
        lidar_features = lidar_features / lidar_features.norm(dim=-1,
                                                              keepdim=True)
        joystick_features = joystick_features / joystick_features.norm(
            dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # batch_size * batch_size
        logits_per_lidar = logit_scale * lidar_features @ joystick_features.t()
        logits_per_joystick = logits_per_lidar.t()
        # avg the cross entropy loss in both dimensions
        gt = torch.arange(
            logits_per_lidar.shape[0]).type_as(logits_per_lidar).long()
        loss_l = F.cross_entropy(logits_per_lidar, gt)
        loss_i = F.cross_entropy(logits_per_joystick, gt)
        loss = (loss_l + loss_i) / 2.0
        return loss


class CLIPSocialNavModel(nn.Module):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self, lidar_encoder: VisionTransformer,
                 joystick_encoder: JoyStickEncoder) -> None:
        super().__init__()
        # initialize encoders and temperature parameter
        self.lidar_encoder = lidar_encoder
        self.joystick_encoder = joystick_encoder

    def forward(self, lidar: Tensor, goals: Tensor,
                joystick: Tensor) -> Tuple[Tensor, Tensor]:
        """ forward pass through the CLIP model, sourced from OpenAI CLIP model

        :param lidar: batch of lidar img stacks (batch_size, output_dim)
        :param joystick: batch of joystick data (batch_size, output_dim)
        :return: lidar features and joystick features (batch_size, output_size)
        """
        lidar_features = self.lidar_encoder.forward(lidar, goals)
        joystick_features = self.joystick_encoder.forward(joystick)

        # L2 normalize features
        lidar_features = lidar_features / lidar_features.norm(dim=-1,
                                                              keepdim=True)
        joystick_features = joystick_features / joystick_features.norm(
            dim=-1, keepdim=True)

        return lidar_features, joystick_features


