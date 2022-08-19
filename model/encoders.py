import math
import warnings

import torch
from torch import nn, Tensor

# Vision Transformer from Facebook DINO with some modifications


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """ Feedforward layer for Attention Block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Attention Layer
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class AttnBlock(nn.Module):
    """ Module containing Attention, LayerNormalization, and a Feedforward Layer
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size: int,
                 embed_dim: int) -> None:
        """ Initialize a PatchEmbedding Layer
        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_size: size of the embedding for a patch (input to the transformer)
        """
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size)**2

        # linearly transform patches
        self.lin_proj = nn.Linear(in_features=self.patch_size**2 *
                                  self.input_channels,
                                  out_features=embed_dim)

    def forward(self, x: Tensor) -> Tensor:

        # batch_size, channels, v slices, h slices, patch_size ** 2
        x = x.unfold(2, self.patch_size,
                     self.patch_size).unfold(3, self.patch_size,
                                             self.patch_size)
        # combine vertical and horizontal slices
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size,
                      self.patch_size)
        # batch_size, num patches p channel, patch_size ** 2, channels
        x = x.movedim(1, -1)
        x = x.flatten(-3)  # 3D patch to 1D patch vector
        x = self.lin_proj.forward(x)  # linear transformation
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self,
                 img_size=240,
                 patch_size=8,
                 input_channels=5,
                 embed_dim=256,
                 output_dim=128,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # convert image into patches and linearly embed
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      input_channels=input_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # BERT class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # learnable patch position encodings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))

        # project goal information from 2 dimensions to embed_dim
        self.goal_proj = nn.Linear(in_features=2, out_features=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Attention Blocks
        self.attn_blocks = nn.ModuleList([
            AttnBlock(dim=embed_dim,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=dpr[i],
                      norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              output_dim) if output_dim > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ initialize weight matrix
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1)

    def prepare_tokens(self, lidar, goals):
        """ patch image and linearly embed

        Args:
            lidar (Tensor): batch of lidar images (batch_size, num_channels, img_size, img_size)
            goals (Tensor): batch of goals (batch_size, 2)
        """
        B, nc, w, h = lidar.shape
        x = self.patch_embed(lidar)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        # concatenate goal tokens
        goals = self.goal_proj(goals).unsqueeze(dim=1)
        x = torch.cat((x, goals), dim=1)

        return self.pos_drop(x)

    def forward(self, lidar, goals):
        """ Forward pass through network

        Args:
            lidar (Tensor): batch of lidar images (batch_size, num_channels, img_size, img_size)
            goals (Tensor): batch of goals (batch_size, 2)
        """
        # create tokens
        x = self.prepare_tokens(lidar, goals)
        # pass through attention blocks
        for blk in self.attn_blocks:
            x = blk(x)
        x = self.norm(x)
        # keep only the CLS token
        x = x[:, 0]
        # pass through MLP head
        return self.head(x)

    def get_last_selfattention(self, lidar, goals):
        """ Return the last attention matrix

        Args:
            lidar (Tensor): batch of lidar images (batch_size, num_channels, img_size, img_size)
            goals (Tensor): batch of goals (batch_size, 2)
        """
        x = self.prepare_tokens(lidar, goals)
        for i, blk in enumerate(self.attn_blocks):
            if i < len(self.attn_blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class JoyStickEncoder(nn.Module):
    """ MLP used to generate feature vectors for joystick input
    """

    def __init__(self, joy_len=300, output_dim=128) -> None:
        """ Initialize an mlp network to encode joystick data

        Args:
            joy_len (int): len of each joystick sequence. Defaults to 300
            output_dim (int): final feature dimenstion. Defaults to 128
            dropout (float): probability of dropping a neuron in dropout layer. Defaults to 0.1
        """
        super().__init__()
        joy_len *= 2
        # two linear transformations
        self.fc1 = nn.Linear(joy_len, joy_len, bias=False)
        self.fc2 = nn.Linear(joy_len, output_dim, bias=False)
        self.fc3 = nn.Linear(output_dim, output_dim)

        # activation, dropout, batch normalize
        self.relu = nn.LeakyReLU()
        self.batch_norm_1 = nn.BatchNorm1d(joy_len)
        self.batch_norm_2 = nn.BatchNorm1d(output_dim)

    def forward(self, joystick: Tensor) -> Tensor:
        """ pass a batch of joystick data through the network

        Args:
            x (Tensor): batch of joystick data (batch_size, joy_len, 2)

        Returns:
            Tensor: feature vectors (batch_size, output_dim)
        """
        out = joystick.flatten(1)

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
