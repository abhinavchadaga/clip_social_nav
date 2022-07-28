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

        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_size: size of the embedding for a patch (input to the transformer)
        """
        super(PatchEmbedding, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # linearly transform patches
        self.lin_proj = nn.Linear(in_features=(self.patch_size ** 2) * self.input_channels,
                                  out_features=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        """ Split a batch of images into patches and linearly embed each patch

        :param x: input tensor (batch_size, channels, img_height, img_width)
        :return: a batch of patch embeddings (batch_size, num_patches, embedding_size)
        """
        # sanity checks
        assert len(x.shape) == 4
        batch_size, num_channels, height, width = x.shape
        assert height == width and height == self.img_size

        # batch_size, channels, v slices, h slices, patch_size ** 2
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size,
                                                                 self.patch_size)
        # combine vertical and horizontal slices
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size, self.patch_size)
        x = x.movedim(1, -1)  # batch_size, num patches p channel, patch_size ** 2, channels
        x = x.flatten(-3)  # 3D patch to 1D patch vector
        x = self.lin_proj.forward(x)  # linear transformation
        return x


class LidarEncoder(nn.Module):
    """ Vision Transformer used to generate lidar feature vector
    """

    def __init__(self, img_size: int, input_channels: int, patch_size=32, embedding_size=1280,
                 output_dim=512) -> None:
        """ Create a LidarEncoder

        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_size: size of the embedding for a patch (input to the transformer)
        :param output_dim: size of the output feature
        """
        super(LidarEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, input_channels=input_channels,
                                          patch_size=patch_size, embedding_size=embedding_size)

        # class token from BERT
        # contains all learned information
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_size)))

        # learnable positional embeddings for each patch
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + self.patch_embed.num_patches, embedding_size))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8,
                                                   activation='gelu', batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6,
                                             norm=self.layer_norm)

        # MLP head, only uses the cls token
        self.mlp_head = nn.Linear(embedding_size, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """ pass batch of images and return a feature vector

        :param x: input tensor (batch_size, num_channels, img_size, img_size)
        :return: lidar feature tensor (batch_size, output_size)
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # turn batch of images into embeddings
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # expand cls token from 1 batch
        x = torch.cat((cls_token, x),
                      dim=1)  # concatenate cls token to beginning of patch embeddings
        x += self.positional_embeddings  # add learnable positional embeddings
        x = self.encoder(x)  # pass input with cls token and positional
        # embeddings through transformer encoder
        cls_token = x[:, 0]  # keep only cls token, discard rest
        x = self.mlp_head(cls_token)  # pass cls token into MLP head
        return x


class JoyStickEncoder(nn.Module):
    """ MLP used to generate feature vectors for joystick input
    """

    def __init__(self, input_dim: int, output_dim=512, dropout=0.2) -> None:
        super(JoyStickEncoder, self).__init__()
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
        # load data
        data_path = './data'
        cprint(f'loading data from {data_path}...\n', 'green')
        dm = CLIPDataModule(data_dir='./data', batch_size=256, num_workers=2)
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
        print(f'lidar_batch max: {torch.max(lidar_batch):.2f}')
        print(f'lidar_batch min: {torch.min(lidar_batch):.2f}')
        cprint(f'lidar_batch shape: {lidar_batch.shape}', 'green')
        img_size = lidar_batch.shape[2]
        input_channels = lidar_batch.shape[1]
        lidar_encoder = LidarEncoder(img_size=img_size, input_channels=input_channels,
                                     output_dim=output_size)
        out: Tensor = lidar_encoder(lidar_batch)
        out = out / out.norm(dim=1, keepdim=True)
        print(out.shape)
        print(f'lidar out max: {torch.max(out):.2f}')
        print(f'lidar out min: {torch.min(out):.2f}')
        cprint(f'elapsed time: {time.time() - start:.2f} s\n', 'green', attrs=['bold'])

        # pass joystick data through encoder
        start = time.time()
        joy_batch = batch[1]
        print(f'joystick batch max: {torch.max(joy_batch):.2f}')
        print(f'joystick batch min: {torch.min(joy_batch):.2f}')
        cprint(f'joy_batch shape: {joy_batch.shape}', 'green')
        input_dim = joy_batch.shape[1] * joy_batch.shape[2]
        joystick_encoder = JoyStickEncoder(input_dim=input_dim, output_dim=output_size)
        joy_out = joystick_encoder(joy_batch)
        joy_out = joy_out / joy_out.norm(dim=1, keepdim=True)
        print(joy_out.shape)
        print(f'joystick out max: {torch.max(joy_out):.2f}')
        print(f'joystick out min: {torch.min(joy_out):.2f}')
        cprint(f'elapsed time: {time.time() - start:.2f} s', 'green', attrs=['bold'])


if __name__ == '__main__':
    main()
