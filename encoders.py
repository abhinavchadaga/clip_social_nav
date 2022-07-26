import time

import torch
from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size=16, embedding_size=768):
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
        print(self.num_patches)

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
    def __init__(self, image_size: int, input_channels: int, output_dim: int, device: str,
                 patch_size=16, embedding_size=768, ):
        super(LidarEncoder, self).__init__()
        super(LidarEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=image_size, input_channels=input_channels,
                                          patch_size=patch_size, embedding_size=embedding_size)

        # class token from BERT
        # contains all learned information
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embedding_size)))

        # learnable positional embeddings for each patch
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embedding_size))

        # transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8,
                                                     activation='gelu', batch_first=True,
                                                     device=device),
            num_layers=6)

        # MLP head, only uses the cls token
        self.mlp_head = nn.Linear(embedding_size, output_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embeddings
        x = self.encoder(x)
        cls_token = x[:, 0]
        x = self.mlp_head(cls_token)
        return x


def main():
    start = time.time()
    device = 'mps'
    x = torch.rand((64, 5, 401, 401)).to(device)
    lidar_encoder = LidarEncoder(x.shape[2], x.shape[1], output_dim=100, device=device)
    out = lidar_encoder(x)
    print(out.shape)
    print(f'elapsed time: {time.time() - start:.2f}')


if __name__ == '__main__':
    main()
