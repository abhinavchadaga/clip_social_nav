import torch
from torch import nn, Tensor
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

    def __init__(self, img_size: int, input_channels: int, patch_size: int,
                 embedding_size: int, nhead: int, dropout: float,
                 activation: str, num_layers: int, output_dim: int) -> None:
        """ Initialize a vision transformer to encode a batch of lidar images

        Args:
            img_size (int): img dimension (assume square)
            input_channels (int): number of channels in the image
            patch_size (int): patch dimension (assume square)
            embedding_size (int): final size of a patch embedding vector
            nhead (int): number of self-attention heads per attention layer
            dropout (float): dropout for transformer layer mlp
            activation (str): activation function for transformer mlp
            num_layers (int): number of encoder layers in the encoder
            output_dim (int): final feature dimension
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=1e-6,
                                                   batch_first=True)

        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_layers,
                                             norm=self.layer_norm)

        # MLP head, only uses the cls token
        self.mlp_head = nn.Sequential(nn.Linear(embedding_size, output_dim),
                                      nn.ReLU(),
                                      nn.Linear(output_dim, output_dim))

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


class JoyStickEncoder(nn.Module):
    """ MLP used to generate feature vectors for joystick input
    """

    def __init__(self, joy_len, output_dim, dropout) -> None:
        """ Initialize an mlp network to encode joystick data

        Args:
            joy_len (_type_): len of each joystick sequence
            output_dim (_type_): final feature dimenstion
            dropout (_type_): probability of dropping a neuron in dropout layer
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


def main():
    # Testing forward pass with a batch from the dataloader

    # create datamodule
    dm = CLIPDataModule(data_path='./data',
                        batch_size=64,
                        num_workers=2,
                        verbose=True)

    dm.setup()

    # create encoders
    lidar_encoder = LidarEncoder(img_size=400,
                                 input_channels=5,
                                 patch_size=16,
                                 embedding_size=128,
                                 nhead=1,
                                 dropout=0.,
                                 activation='gelu',
                                 num_layers=3,
                                 output_dim=128)
    joystick_encoder = JoyStickEncoder(joy_len=300, output_dim=128, dropout=0.)

    # load a batch from the training set
    batch = next(iter(dm.train_dataloader()))
    lidar, joystick, goal = batch

    # print batch shapes
    print('')
    cprint('batch shapes\t\t', 'white', attrs=['bold'])
    print('lidar:\t\t', lidar.shape)
    print('joystick:\t', joystick.shape)
    print('goal:\t\t', goal.shape)
    print('')

    # pass batch through encoders
    lidar_feature = lidar_encoder.forward(lidar, goal)
    joy_feature = joystick_encoder.forward(joystick)

    # print feature shape information
    cprint('feature shapes', 'white', attrs=['bold'])
    print('lidar_feature:\t\t', lidar_feature.shape)
    print('joystick_feature:\t', joy_feature.shape)


if __name__ == '__main__':
    main()
