from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from encoders import LidarEncoder, JoyStickEncoder


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, lidar_features, joystick_features):
        assert lidar_features.shape == joystick_features.shape

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # batch_size * batch_size
        logits = logit_scale * (lidar_features @ joystick_features.t())

        # avg the cross entropy loss in both dimensions
        labels = torch.arange(logits.shape[0])
        loss_l = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.t(), labels.t())
        loss = (loss_l + loss_i) / 2.0
        return loss


class CLIPSocialNavModel(pl.LightningModule):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self, l_img_size=401, l_input_channels=5, l_patch_size=32, l_embedding_size=1280,
                 j_input_dim=900, j_dropout=0., output_dim=512, lr=3e-5, weight_decay=1e-5,
                 temperature=0.7) -> None:
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

        self.loss = CLIPLoss(temperature=temperature)

        # optimizer parameters
        self.learning_rate = lr
        self.weight_decay = weight_decay

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
        lidar_features = lidar_features / lidar_features.norm(dim=-1, keepdim=True)
        joystick_features = joystick_features / joystick_features.norm(dim=-1, keepdim=True)

        return lidar_features, joystick_features

    def configure_optimizers(self):
        """ Setup optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode="min",
                                                                  patience=2,
                                                                  factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'validation loss'
        }

    def training_step(self, batch, batch_idx):
        """ Forward pass and compute training loss for one step
        """
        lidar, joystick = batch
        lidar_features, joystick_features = self.forward(lidar.float(), joystick.float())
        loss = self.loss(lidar_features, joystick_features)
        self.log('training_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lidar, joystick = batch
        lidar_features, joystick_features = self.forward(lidar, joystick)
        loss = self.loss(lidar_features, joystick_features)
        self.log('validation_loss', loss, prog_bar=True, logger=True)
        return loss
