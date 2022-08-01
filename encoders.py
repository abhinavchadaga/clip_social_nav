import time

import torch
from termcolor import cprint
from torch import nn, Tensor

from dataset import CLIPDataModule


class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size=32, embedding_size=1280) -> \
            None:
        """ Initialize a PatchEmbedding Layer

        Args:
            img_size: size of the input images (assume square)
            input_channels: number of input channels
            patch_size: size of a 2d patch (assume square)
            embedding_size: size of the embedding vector for a patch (input to the transformer)
        """
        super(PatchEmbedding, self).__init__()
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
        """ Split a batch of images into patches and linearly embed each patch

        :param lidar_batch: input tensor (batch_size, channels, img_height, img_width)
        :return: a batch of patch embeddings (batch_size, num_patches, embedding_size)
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
                 img_size: int,
                 input_channels: int,
                 patch_size=32,
                 embedding_size=1280,
                 output_dim=512,
                 msa_heads=8,
                 activation='gelu',
                 num_layers=6) -> None:
        """ Create a LidarEncoder

        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_size: size of the embedding for a patch (input to the transformer)
        :param output_dim: size of the output feature
        """
        super(LidarEncoder, self).__init__()
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
                                                   nhead=msa_heads,
                                                   activation=activation,
                                                   batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_layers,
                                             norm=self.layer_norm)

        # MLP head, only uses the cls token
        self.mlp_head = nn.Linear(embedding_size, output_dim)

    def forward(self, lidar_patches: Tensor, goals: Tensor) -> Tensor:
        """ pass batch of images and goals and return a feature vector

        Args:
            lidar_patches: batch of lidar images (batch_size, num_patches, embedding size)
            goals: batch of goals (batch_size, 2)

        Returns:
            feature vector of size (batch_size, output_dim)

        """
        batch_size = lidar_patches.shape[0]
        # turn batch of images into embeddings
        out = self.patch_embed(lidar_patches)
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

    def __init__(self, input_dim=300, output_dim=512, dropout=0.) -> None:
        super(JoyStickEncoder, self).__init__()
        input_dim *= 3
        # two linear transformations
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

        # activation, dropout, batch normalize
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)

        # Input to hidden layer
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        # hidden layer to output
        x = self.batch_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


def main():
    with torch.no_grad():
        data_path = './data'
        dm = CLIPDataModule(data_dir=data_path,
                            batch_size=256,
                            num_workers=2,
                            future_joy_len=500,
                            verbose=True)
        dm.setup()

        cprint('creating trainloader...\n', 'green')
        # get a random batch from training set
        trainloader = dm.train_dataloader()
        train_iter = iter(trainloader)
        batch = next(train_iter)
        cprint('done creating trainloader\n', 'green')

        output_size = 512
        cprint(f'output_size: {output_size}\n', 'cyan', attrs=['bold'])

        # pass lidar data through encoder
        start = time.time()
        lidar_batch = batch[0]
        relative_goals = batch[2]
        print(f'lidar_batch max: {torch.max(lidar_batch):.2f}')
        print(f'lidar_batch min: {torch.min(lidar_batch):.2f}')
        cprint(f'lidar_batch shape: {lidar_batch.shape}', 'green')
        img_size = lidar_batch.shape[2]
        input_channels = lidar_batch.shape[1]
        lidar_encoder = LidarEncoder(img_size=img_size,
                                     input_channels=input_channels,
                                     output_dim=output_size,
                                     msa_heads=4,
                                     num_layers=3)

        out: Tensor = lidar_encoder(lidar_batch, relative_goals)
        out = out / out.norm(dim=1, keepdim=True)
        print(out.shape)
        print(f'lidar out max: {torch.max(out):.2f}')
        print(f'lidar out min: {torch.min(out):.2f}')
        cprint(f'elapsed time: {time.time() - start:.2f} s\n',
               'green',
               attrs=['bold'])

        # pass joystick data through encoder
        start = time.time()
        joy_batch = batch[1]
        print(f'joystick batch max: {torch.max(joy_batch):.2f}')
        print(f'joystick batch min: {torch.min(joy_batch):.2f}')
        cprint(f'joy_batch shape: {joy_batch.shape}', 'green')
        input_dim = joy_batch.shape[1]
        joystick_encoder = JoyStickEncoder(input_dim=input_dim,
                                           output_dim=output_size)
        joy_out = joystick_encoder(joy_batch)
        joy_out = joy_out / joy_out.norm(dim=1, keepdim=True)
        print(joy_out.shape)
        print(f'joystick out max: {torch.max(joy_out):.2f}')
        print(f'joystick out min: {torch.min(joy_out):.2f}')
        cprint(f'elapsed time: {time.time() - start:.2f} s',
               'green',
               attrs=['bold'])


if __name__ == '__main__':
    main()
