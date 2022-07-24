import torch
from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size=16, embedding_dim=768):
        """ Initialize a PatchEmbedding Layer

        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_dim: size of the embedding for a patch (input to the transformer)
        """
        super(PatchEmbedding, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size

        # linearly transform patches
        self.lin_proj = nn.Linear(in_features=self.patch_size ** 2 * self.input_channels,
                                  out_features=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        # sanity checks
        assert len(x.shape) == 4
        batch_size, num_channels, height, width = x.shape
        assert height == width and height == self.img_size

        # batch_size, channels, v slices, h slices, patch_size, patch_size
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size,
                                                                 self.patch_size)
        # batch_size, channels, num_patches per channel, patch_size, patch_size
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size, self.patch_size)
        # batch_size, num_patches per channel, patch_size, patch_size, channels
        x = x.movedim(1, -1)
        x = x.flatten(-3)  # 2D patch to 1D patch vector
        x = self.lin_proj.forward(x)  # linear transformation
        return x


def main():
    img = torch.rand((1, 5, 401, 401))
    patcher = PatchEmbedding(img_size=img.shape[2], input_channels=img.shape[1])
    patches = patcher.forward(img)
    print(patches.shape)


if __name__ == "__main__":
    main()
