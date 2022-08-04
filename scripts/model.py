import math
import time
from typing import Tuple, Union
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from termcolor import cprint

from dataset import CLIPDataModule
from vision_transformer import DINOEncoder
from encoders import JoyStickEncoder, LidarEncoder
from vit_lucidrains import ViT


class CLIPLoss(pl.LightningModule):
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
        gt = torch.arange(logits_per_lidar.shape[0],
                          dtype=torch.long,
                          device=self.device)
        loss_l = F.cross_entropy(logits_per_lidar, gt)
        loss_i = F.cross_entropy(logits_per_joystick, gt)
        loss = (loss_l + loss_i) / 2.0
        return loss


class CLIPSocialNavModel(pl.LightningModule):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self,
                 lidar_encoder: Union[DINOEncoder, ViT],
                 joystick_encoder: JoyStickEncoder,
                 temperature=0.07,
                 lr=3e-5,
                 weight_decay=1e-5) -> None:
        super().__init__()
        # initialize encoders and temperature parameter
        self.lidar_encoder = lidar_encoder
        self.joystick_encoder = joystick_encoder
        self.clip_loss = CLIPLoss(temperature=temperature)
        # optimizer parameters
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def forward(self, lidar: Tensor, joystick: Tensor,
                goals: Tensor) -> Tuple[Tensor, Tensor]:
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

    def configure_optimizers(self):
        """ Setup optimizer and learning rate scheduler
        """
        # weight decay, ignore the biases
        weights = [
            param for name, param in self.named_parameters()
            if 'bias' not in name
        ]

        optimizer = torch.optim.AdamW(params=weights,
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      amsgrad=True)
        scheduler = cosine_scheduler()

        return {'optimizer': optimizer, 'monitor': 'validation_loss'}

    def training_step(self, batch, batch_idx):
        """ Forward pass and compute training loss for one step
        """
        lidar, joystick, goals = batch
        lidar_features, joystick_features = self.forward(
            lidar, joystick, goals)
        return {'lidar': lidar_features, 'joystick': joystick_features}

    def training_step_end(self, step_output):
        lidar_features = step_output['lidar']
        joystick_features = step_output['joystick']
        loss = self.clip_loss(lidar_features, joystick_features)
        self.log("training_loss",
                 loss,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Forward pass and compute training loss for one step
        """
        lidar, joystick, goals = batch
        lidar_features, joystick_features = self.forward(
            lidar, joystick, goals)
        return {'lidar': lidar_features, 'joystick': joystick_features}

    def validation_step_end(self, step_output):
        lidar_features = step_output['lidar']
        joystick_features = step_output['joystick']
        loss = self.clip_loss(lidar_features, joystick_features)
        self.log("validation_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
