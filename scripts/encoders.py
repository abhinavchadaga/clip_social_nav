from matplotlib.colors import Normalize
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from termcolor import cprint

from dataset import CLIPDataModule


class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size: int,
                 embedding_size: int) -> None:
        """ Initialize a patch embedding layer

        Args:
            img_size (int): img dimension (assume square)
            input_channels (int): number of channels in the image
            patch_size (int): patch dimension (assume square)
            embedding_size (int): final size of a patch embedding vector
        """
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size)**2

        # # projection layer for the goal patch
        # self.goal_proj = nn.Linear(in_features=2, out_features=embedding_size)

        # linearly transform patches
        self.img_patch_proj = nn.Linear(in_features=(self.patch_size**2) *
                                        self.input_channels,
                                        out_features=embedding_size)

    def forward(self, lidar_batch: Tensor) -> Tensor:
        """ split each image in batch into patches and linearly embed the patches

        Args:
            lidar_batch (Tensor): batch of lidar images
                (batch_size, num_channels, img_size, img_size)

        Returns:
            Tensor: lidar patches (batch_size, num_patches, embedding_size)
        """
        # sanity checks
        assert len(lidar_batch.shape) == 4
        batch_size, num_channels, height, width = lidar_batch.shape
        assert height == width and height == self.img_size

        # batch_size, channels, v slices, h slices, patch_size ** 2
        lidar_batch = lidar_batch.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)
        # combine vertical and horizontal slices
        lidar_batch = lidar_batch.reshape(lidar_batch.shape[0],
                                          lidar_batch.shape[1], -1,
                                          self.patch_size, self.patch_size)
        # batch_size, num patches per channel, patch_size ** 2, channels
        lidar_batch = lidar_batch.movedim(1, -1)
        # 3D patch to 1D patch vector
        lidar_batch = lidar_batch.flatten(-3)
        # linear transformation
        out = self.img_patch_proj(lidar_batch)
        return out


class LidarEncoder(nn.Module):
    """ Vision Transformer used to generate lidar feature vector
    """

    def __init__(self,
                 img_size=400,
                 input_channels=5,
                 patch_size=16,
                 embedding_size=128,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.1,
                 activation='gelu',
                 num_layers=6,
                 output_dim=100) -> None:
        """ Initialize a vision transformer to encode a batch of lidar images

        Args:
            img_size (int, optional): img dimension (assume square).
                Defaults to 400.
            input_channels (int, optional): number of channels in the image.
                Defaults to 5.
            patch_size (int, optional): patch dimension (assume square).
                Defaults to 16.
            embedding_size (int, optional): final size of a patch embedding vector.
                Defaults to 128.
            dim_feedforward (int, optional): the dimension of the feedforward network model.
                Defaults to 2048.
            nhead (int, optional): number of self-attention heads per attention layer.
                Defaults to 1.
            dropout (_type_, optional): dropout for transformer layer mlp.
                Defaults to 0.1
            activation (str, optional): activation function for transformer mlp.
                Defaults to 'gelu'.
            num_layers (int, optional): number of encoder layers in the encoder.
                Defaults to 3.
            output_dim (int, optional): final feature dimension.
                Defaults to 128.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size,
                                          input_channels=input_channels,
                                          patch_size=patch_size,
                                          embedding_size=embedding_size)

        # class token from BERT, contains all learned information
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_size)))

        # project the goal tokens to embedding size
        self.goal_proj = nn.Linear(in_features=2, out_features=embedding_size)

        # learnable positional embeddings for each patch
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + self.patch_embed.num_patches, embedding_size))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-6,
            batch_first=True)

        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_layers,
                                             norm=self.layer_norm)

        # MLP head, only uses the cls token
        self.mlp_head = MLPHead(input_dim=embedding_size,
                                hidden_dim=embedding_size,
                                output_dim=output_dim)

    def forward(self, lidar: Tensor, goals: Tensor) -> Tensor:
        """ pass a batch of lidar images and goal points through the lidar encoder

        Args:
            lidar (Tensor): batch of lidar images (batch_size, num_channels, img_size, img_size)
            goals (Tensor): batch of goals (batch_size, 2)

        Returns:
            Tensor: encoder feature vector (batch_size, output_dim)
        """
        batch_size = lidar.shape[0]
        # turn batch of images into embeddings
        out = self.patch_embed(lidar)
        # expand cls token from 1 batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # concatenate cls token to beginning of patch embeddings
        out = torch.cat((cls_token, out), dim=1)
        # add learnable positional embeddings
        out += self.positional_embeddings
        # project goals from (batch_size, 2) to (batch_size, 1, embedding_size)
        goals = self.goal_proj(goals).unsqueeze(dim=1)
        # concatenate goal tokens
        out = torch.cat((out, goals), dim=1)
        # pass input with cls token and positional embeddings through transformer encoder
        out = self.encoder(out)
        # keep only cls token, discard rest
        out = out[:, 0]
        # pass cls token into MLP head
        out = self.mlp_head(out)
        return out


class MLPHead(nn.Module):
    """ MLP head for lidar vision transformer
    """

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class JoyStickEncoder(nn.Module):
    """ MLP used to generate feature vectors for joystick input
    """

    def __init__(self, joy_len=300, output_dim=100, dropout=0.1) -> None:
        """ Initialize an mlp network to encode joystick data

        Args:
            joy_len (int): len of each joystick sequence. Defaults to 300
            output_dim (int): final feature dimenstion. Defaults to 128
            dropout (float): probability of dropping a neuron in dropout layer. Defaults to 0.1
        """
        super().__init__()
        joy_len *= 3
        # two linear transformations
        self.fc1 = nn.Linear(joy_len, joy_len, bias=False)
        self.fc2 = nn.Linear(joy_len, output_dim, bias=False)
        self.fc3 = nn.Linear(output_dim, output_dim)

        # activation, dropout, batch normalize
        self.relu = nn.LeakyReLU()
        self.batch_norm_1 = nn.BatchNorm1d(joy_len)
        self.batch_norm_2 = nn.BatchNorm1d(output_dim)

    def forward(self, joystick_batch: Tensor) -> Tensor:
        """ pass a batch of joystick data through the network

        Args:
            x (Tensor): batch of joystick data (batch_size, joy_len, 3)

        Returns:
            Tensor: feature vectors (batch_size, output_dim)
        """
        out = joystick_batch.flatten(1)

        # Input to hidden layer
        out = self.fc1(out)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        # hidden1 to hidden2
        out = self.fc2(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)

        # hidden layer 2 to output
        out = self.fc3(out)
        return out