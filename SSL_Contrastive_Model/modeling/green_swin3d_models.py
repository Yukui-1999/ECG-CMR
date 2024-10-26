# Modified by Lang Huang (laynehuang@outlook.com)
# All rights reserved.
from operator import mul

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Swin Transformer: https://github.com/microsoft/Swin-Transformer
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from functools import reduce, lru_cache
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from modeling.base_green_models import BaseGreenModel3D
from modeling.group_window_attention3d import WindowAttention3D, GroupingModule3D, get_coordinates3d


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, rel_pos_idx):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        x = self.attn(x, mask_matrix, rel_pos_idx)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging3D(nn.Module):
    r""" Patch Merging Layer for 3D data.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, coords_prev, mask_prev):
        """
        x: B, D*H*W, C
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        # gather patches lie within 2x2x2 local window
        mask = mask_prev.reshape(D // 2, 2, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1)
        coords = get_coordinates3d(D, H, W, device=x.device).reshape(3, -1).permute(1, 0)
        coords = coords.reshape(D // 2, 2, H // 2, 2, W // 2, 2, 3).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, 3)
        coords_vis_local = coords[mask].reshape(-1, 3)
        coords_vis_local = coords_vis_local[:, 0] * H * W + coords_vis_local[:, 1] * W + coords_vis_local[:, 2]
        idx_shuffle = torch.argsort(torch.argsort(coords_vis_local))
        x = torch.index_select(x, 1, index=idx_shuffle)

        x = x.reshape(B, L // 8, 8, C)
        # row-first order to column-first order
        x = torch.cat([
            x[:, :, 0], x[:, :, 4], x[:, :, 2], x[:, :, 6],
            x[:, :, 1], x[:, :, 5], x[:, :, 3], x[:, :, 7]
        ], dim=-1)

        # x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        # x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        # x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        # x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        # merging by a linear layer
        x = self.norm(x)
        x = self.reduction(x)

        mask_new = mask_prev.view(1, D // 2, 2, H // 2, 2, W // 2, 2).sum(dim=(2, 4, 6))
        assert torch.unique(mask_new).shape[0] == 2
        mask_new = (mask_new > 0).reshape(1, -1)
        coords_new = get_coordinates3d(D // 2, H // 2, W // 2, x.device).reshape(1, 3, -1)
        coords_new = coords_new.transpose(2, 1)[mask_new].reshape(1, -1, 3)
        return x, coords_new, mask_new

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer3D(nn.Module):
    """ A basic Swin Transformer layer for one stage for 3D data.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (D, H, W).
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (D, H, W).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(2, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, coords, patch_mask):
        # prepare the attention mask and relative position bias
        group_block = GroupingModule3D(self.window_size, shift_size=(0, 0, 0))
        mask, pos_idx = group_block.prepare(coords, num_tokens=x.shape[1])
        if max(self.window_size) < min(self.input_resolution):
            group_block_shift = GroupingModule3D(self.window_size, tuple(i // 2 for i in self.window_size))
            mask_shift, pos_idx_shift = group_block_shift.prepare(coords, num_tokens=x.shape[1])
        else:
            # do not shift
            group_block_shift = group_block
            mask_shift, pos_idx_shift = mask, pos_idx

        # forward with grouping/masking
        for i, blk in enumerate(self.blocks):
            gblk = group_block if i % 2 == 0 else group_block_shift
            attn_mask = mask if i % 2 == 0 else mask_shift
            rel_pos_idx = pos_idx if i % 2 == 0 else pos_idx_shift
            x = gblk.group(x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, rel_pos_idx)
            else:
                x = blk(x, attn_mask, rel_pos_idx)
            x = gblk.merge(x)

        # patch merging
        if self.downsample is not None:
            x, coords, patch_mask = self.downsample(x, coords, patch_mask)

        return x, coords, patch_mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# patchEmbedding3D
class PatchEmbed3D(nn.Module):
    """ Patch Embedding for 3d images.
    Args:
        input_resolution (tuple[int]): Resolution of input feature (input 3D image).
        patch_size (int): Patch token size. Default: (2, 4, 4).
        in_chans (int): Number of input channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, input_resolution=(64, 224, 224), patch_size=(2, 4, 4), in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # self.patches_resolution = [input_resolution[i] // patch_size[i] for i in range(3)]
        # self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)  # B C D Wh Ww

        x = x.flatten(2).transpose(1, 2)  # B Wh*Ww*D C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer3D(BaseGreenModel3D):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple(int)): Window size. Default: (2, 7, 7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, image_resolution=(64, 224, 224), patch_size=(4, 4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):

        super().__init__()

        self.input_resolution = image_resolution
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape  # absolute position embedding
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate

        assert all([i % j == 0 for i, j in zip(image_resolution, patch_size)]), \
            "image resolution must be divisible by patch size"

        # patch resolution
        self.patches_resolution = [image_resolution[0] // patch_size[0],
                                   image_resolution[1] // patch_size[1],
                                   image_resolution[2] // patch_size[2]]
        # number of patches
        num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            input_resolution=image_resolution, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer),
                                  self.patches_resolution[2] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def patchify(self, x):
        # patch embedding
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def forward_features(self, x, mask):
        b, c, d, h, w = x.shape
        # patch embedding
        x = self.patchify(x)

        # mask out some patches according to the random mask
        x_vis, coords, vis_mask = self.apply_mask(x, d, h, w, mask, self.patches_resolution)

        # transformer forward
        for layer in self.layers:
            x_vis, coords, vis_mask = layer(x_vis, coords, vis_mask)
        x_vis = self.norm(x_vis)

        return x_vis


if __name__ == '__main__':
    '''Unit test for the PatchMerging module'''
    # resolution of the final stage
    D, H, W = 4, 5, 5
    mask_ratio = 3 / 5.
    nvis, nmasked = int(D * H * W * (1 - mask_ratio)), int(D * H * W * mask_ratio)
    mask_new = torch.cat([torch.ones((nvis,)), torch.zeros((nmasked,))])
    mask_new = mask_new[torch.randperm(D * H * W)].bool()

    # the second last stage
    mask = mask_new.reshape(D, 1, H, 1, W, 1).repeat((1, 2, 1, 2, 1, 2)).reshape(-1)  # D*H*W*8
    D, H, W = D * 2, H * 2, W * 2
    x_ori = torch.arange(D * H * W) * 100.
    x = x_ori[mask].reshape(1, nvis * 8, -1)  # [1 ,nvis * 8, C]
    B, L, C = x.shape
    print(mask.reshape(D, H, W).shape)
    print(x_ori.reshape(D, H, W).shape)
    print(x.shape)

    mask = mask.reshape(D // 2, 2, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1)  # D*H*W, 8
    coords = get_coordinates3d(D, H, W).reshape(3, -1).permute(1, 0)  # D*H*W, 3
    coords = coords.reshape(D // 2, 2, H // 2, 2, W // 2, 2, 3).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, 3)  # D*H*W, 3
    coords_vis_local = coords[mask].reshape(-1, 3)
    coords_vis_local = coords_vis_local[:, 0] * H * W + coords_vis_local[:, 1] * W + coords_vis_local[:, 2]
    idx_shuffle = torch.argsort(torch.argsort(coords_vis_local))

    # shuffle
    x = torch.index_select(x, 1, index=idx_shuffle)
    x = x.reshape(B, L // 8, 8 * C).squeeze()
    print(x.squeeze().shape)
