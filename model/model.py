import time
from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pytorch_lightning as pl

from encoders import VisionTransformer, JoyStickEncoder
import config as CFG


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


class CLIPSocialNavModel(pl.LightningModule):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self, lidar_encoder: VisionTransformer,
                 joystick_encoder: JoyStickEncoder, temperature: int) -> None:
        super().__init__()
        # initialize encoders and temperature parameter
        self.lidar_encoder = lidar_encoder
        self.joystick_encoder = joystick_encoder
        self.clip_loss = CLIPLoss(temperature=temperature)

    def forward(self, lidar: Tensor, goals: Tensor,
                joystick: Tensor) -> Tuple[Tensor, Tensor]:
        """ forward pass through the CLIP model, sourced from OpenAI CLIP model

        Args:
            lidar (Tensor): batch of lidar img stacks (batch_size, output_dim)
            joystick (Tensor): batch of joystick data (batch_size, output_dim)

        Return:
            Tuple[Tensor, Tensor]: lidar features and joystick features (batch_size, output_size)
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
        weights = [
            param for name, param in self.named_parameters()
            if 'bias' not in name
        ]

        optimizer = torch.optim.AdamW(params=weights,
                                      lr=CFG.learning_rate,
                                      weight_decay=CFG.weight_decay,
                                      amsgrad=True)
        return optimizer

    def _get_batch_similarity(self, joystick_batch: torch.Tensor) -> float:
        m = torch.empty((joystick_batch.shape[0], joystick_batch.shape[0]))
        joystick_batch = joystick_batch.flatten(1)
        for x in range(m.shape[0]):
            x_1 = joystick_batch[x, :]
            for y in range(m.shape[0]):
                x_2 = joystick_batch[y, :]
                m[x, y] = torch.linalg.norm(x_2 - x_1)
        similarity = torch.sum(m).item() / m.shape[0]**2
        return similarity

    def on_train_batch_start(self, batch, batch_idx) -> None:
        _, _, joystick = batch
        self.train_batch_sim = self._get_batch_similarity(joystick)

    def training_step(self, batch, batch_idx):
        """ Forward pass through network for training

        Return:
            dict: lidar and joystick feature vectors, 
        """
        lidar, goals, joystick = batch
        lidar_features, joystick_features = self.forward(
            lidar, goals, joystick)
        return {'lidar': lidar_features, 'joystick': joystick_features}

    def training_step_end(self, step_output):
        """ Merge feature vectors across GPUs and calculate training loss
        """

        lidar_features = step_output['lidar']
        joystick_features = step_output['joystick']
        self.log("train_batch_sim",
                 self.train_batch_sim,
                 prog_bar=True,
                 logger=False,
                 on_step=True,
                 on_epoch=False)
        if self.train_batch_sim <= CFG.sim_threshold:
            return None
        else:
            loss = self.clip_loss(lidar_features, joystick_features)
            self.log("training_loss",
                     loss,
                     on_epoch=True,
                     on_step=True,
                     prog_bar=True,
                     logger=True)
            return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # clear out batch_similarity
        self.train_batch_sim = None

    def on_validation_batch_start(self, batch, batch_idx,
                                  dataloader_idx) -> None:
        _, _, joystick = batch
        self.val_batch_sim = self._get_batch_similarity(joystick)

    def validation_step(self, batch, batch_idx):
        """ Forward pass through network for validation
        Same as training step
        """
        return self.training_step(batch, batch_idx)

    def validation_step_end(self, step_output):
        """ Merge feature vectors across GPUs and calculate validation loss
        """
        if step_output is None:
            return None
        lidar_features = step_output['lidar']
        joystick_features = step_output['joystick']
        if self.val_batch_sim <= CFG.sim_threshold:
            loss = 0
        else:
            loss = self.clip_loss(lidar_features, joystick_features)
            self.log("validation_loss",
                     loss,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx,
                                dataloader_idx):
        # clear out batch_similarity
        self.val_batch_sim = None
