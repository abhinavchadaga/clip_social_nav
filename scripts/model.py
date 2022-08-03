import time
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from termcolor import cprint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from dataset import CLIPDataModule


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(dim,
                                  heads=heads,
                                  dim_head=dim_head,
                                  dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 num_classes,
                 embedding_size,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width //
                                                        patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height,
                      p2=patch_width),
            nn.Linear(patch_dim, embedding_size),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.goal_proj = nn.Linear(in_features=2, out_features=embedding_size)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embedding_size, depth, heads, dim_head,
                                       mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(embedding_size),
                                      nn.Linear(embedding_size, num_classes))

    def forward(self, img, goal):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        goal = self.goal_proj(goal).unsqueeze(dim=1)
        x = torch.cat((x, goal), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class JoyStickEncoder(nn.Module):
    """ MLP used to generate feature vectors for joystick input
    """

    def __init__(self, joy_len=300, output_dim=128, dropout=0.1) -> None:
        """ Initialize an mlp network to encode joystick data

        Args:
            joy_len (_type_): len of each joystick sequence. Defaults to 300
            output_dim (_type_): final feature dimenstion. Defaults to 128
            dropout (_type_): probability of dropping a neuron in dropout layer. Defaults to 0.1
        """
        super().__init__()
        joy_len *= 3
        # two linear transformations
        self.fc1 = nn.Linear(joy_len, joy_len)
        self.fc2 = nn.Linear(joy_len, output_dim)

        # activation, dropout, batch normalize
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(joy_len)

    def forward(self, joystick_batch: Tensor) -> Tensor:
        """ pass a batch of joystick data through the network

        Args:
            x (Tensor): batch of joystick data (batch_size, joy_len, 3)

        Returns:
            Tensor: feature vectors (batch_size, output_dim)
        """
        out = joystick_batch.flatten(1)

        # Input to hidden layer
        out = self.batch_norm(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)

        # hidden layer to output
        out = self.batch_norm(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        return out


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

    def training_step(self, batch, batch_idx):
        """ Forward pass and compute training loss for one step
        """
        lidar, joystick, goals = batch
        lidar_features, joystick_features = self.forward(
            lidar, joystick, goals)

        loss = self.clip_loss(lidar_features, joystick_features)
        self.log('training_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
    model = CLIPSocialNavModel(lidar_encoder=ViT(image_size=400,
                                                 patch_size=16,
                                                 num_classes=128,
                                                 embedding_size=128,
                                                 depth=6,
                                                 heads=16,
                                                 mlp_dim=256,
                                                 channels=5,
                                                 dropout=0.1,
                                                 emb_dropout=0.1,
                                                 pool='cls'),
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