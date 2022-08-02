import time
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from termcolor import cprint

from dataset import CLIPDataModule
from encoders import LidarEncoder, JoyStickEncoder


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
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, lidar_features, joystick_features) -> float:
        """ calculate the loss given a set of lidar feature vectors and joystick feature vectors

        Args:
            lidar_features (_type_): output of lidar_encoder (batch_size, output_dim)
            joystick_features (_type_): output of joystick encoder (batch_size, output_dim)

        Returns:
            float: total loss
        """
        assert lidar_features.shape == joystick_features.shape

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # batch_size * batch_size
        logits = logit_scale * (lidar_features @ joystick_features.t())

        # avg the cross entropy loss in both dimensions
        labels = torch.arange(logits.shape[0], device=self.device)
        loss_l = self.cross_entropy(logits, labels)
        loss_i = self.cross_entropy(logits.t(), labels.t())
        loss = (loss_l + loss_i) / 2.0
        return loss


class CLIPSocialNavModel(pl.LightningModule):
    """ Dual Encoder CLIP Model for learning socially compliant features
    """

    def __init__(self,
                 lidar_encoder,
                 joystick_encoder,
                 temperature=0.07,
                 lr=3e-5,
                 weight_decay=1e-5) -> None:
        super().__init__()
        # create encoders
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
        :return: concatenated tensor containing lidar and joystick features (batch_size * 2,
        output_size)
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
                                      weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode="min",
                                                                  patience=2,
                                                                  factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'validation_loss'
        }

    def training_step(self, batch):
        """ Forward pass and compute training loss for one step
        """
        lidar, joystick, goals = batch
        lidar_features, joystick_features = self.forward(
            lidar, joystick, goals)

        loss = self.clip_loss(lidar_features, joystick_features)
        self.log('training_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        lidar, joystick, goals = batch
        lidar_features, joystick_features = self.forward(
            lidar, joystick, goals)
        loss = self.clip_loss(lidar_features, joystick_features)
        self.log('validation_loss', loss, prog_bar=True, logger=True)
        return loss


def main():
    # test training step
    # create datamodule
    dm = CLIPDataModule(data_path='./data',
                        batch_size=64,
                        num_workers=2,
                        verbose=True)

    dm.setup()

    # load a batch from the training set
    batch = next(iter(dm.train_dataloader()))
    lidar, joystick, goals = batch

    # create model
    model = CLIPSocialNavModel(lidar_encoder=LidarEncoder(),
                               joystick_encoder=JoyStickEncoder(),
                               lr=3e-5,
                               weight_decay=1e-5)
    clip_loss = CLIPLoss()
    start = time.time()
    lidar_features, joystick_features = model(lidar, joystick, goals)
    loss = clip_loss(lidar_features, joystick_features)
    cprint(f'loss: {loss:.2f}', 'white', attrs=['bold'])
    cprint(f'success! 1 step in {time.time() - start:.2f} seconds',
           'green',
           attrs=['bold'])


if __name__ == "__main__":
    main()