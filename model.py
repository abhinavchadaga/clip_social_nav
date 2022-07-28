from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from encoders import LidarEncoder, JoyStickEncoder


class CLIPSocialNavModel(pl.LightningModule):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self, l_img_size=401, l_input_channels=5, l_patch_size=32, l_embedding_size=1280,
                 j_input_dim=900, j_dropout=0.2, output_dim=512, include_logit_scale=True) -> None:
        super(CLIPSocialNavModel, self).__init__()
        # create encoders
        self.lidar_encoder = LidarEncoder(img_size=l_img_size,
                                          input_channels=l_input_channels,
                                          patch_size=l_patch_size,
                                          embedding_size=l_embedding_size,
                                          output_dim=output_dim)
        self.joystick_encoder = JoyStickEncoder(input_dim=j_input_dim,
                                                output_dim=output_dim,
                                                dropout=j_dropout)
        if include_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, lidar: Tensor, joystick: Tensor) -> Tuple[Tensor, Tensor]:
        """ forward pass through the CLIP model, sourced from OpenAI CLIP model

        :param lidar: batch of lidar img stacks (batch_size, output_dim)
        :param joystick: batch of joystick data (batch_size, output_dim)
        :return: concatenated tensor containing lidar and joystick features (batch_size * 2,
        output_size)
        """
        lidar_features = self.lidar_encoder(lidar)
        joystick_features = self.joystick_encoder(joystick)

        # L2 normalize features
        lidar_features = lidar_features / \
            lidar_features.norm(dim=-1, keepdim=True)
        joystick_features = joystick_features / \
            joystick_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_lidar = logit_scale * \
            (lidar_features @ joystick_features.t())  # batch_size *
        # batch_size
        logits_per_joystick = logits_per_lidar.t()

        return logits_per_lidar, logits_per_joystick

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=1000)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        lidar, joystick = batch[0], batch[1]
        logits, _ = self.forward(lidar, joystick)
        labels = torch.arange(lidar.shape[0])
        loss_l = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.t(), labels.t())
        loss = (loss_l + loss_i) / 2.0
        self.log("training loss", loss)
        return loss
