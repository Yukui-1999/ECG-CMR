# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, pred_metric=False,num_classes=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.embed_dim, num_classes)
        self.global_pool = global_pool
        if pred_metric:
            print(f'pred_metric: yes')
            self.head = nn.Sequential(
                    nn.Linear(self.embed_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 82)
                )
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        self.latent = x
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1).unsqueeze(1)  # global pool without cls token .unsqueeze(1)
            outcome = self.fc_norm(x)
        else:
            # x = self.norm(x) # 2024.10.10
            outcome = x[:, 0]

        return outcome
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        
        
        
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# orig
# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         img_size=96, patch_size=16, in_chans=50, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=96, patch_size=16, in_chans=50, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit3d_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(50, 96, 96), patch_size=(10, 16, 16), in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  embed_layer=PatchEmbed3D, **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_3tuple
from timm.layers.trace_utils import _assert

_logger = logging.getLogger(__name__)


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Union[int, tuple] = 224, # D, H, W
            patch_size: Union[int, tuple] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = to_3tuple(patch_size)
        else:
             self.patch_size = patch_size
        if img_size is not None:
            if isinstance(img_size, int):
                self.img_size = to_3tuple(img_size)
            else:
                self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(D == self.img_size[0], f"Input depth ({D}) doesn't match model ({self.img_size[0]}).")
                _assert(H == self.img_size[1], f"Input hight ({H}) doesn't match model ({self.img_size[1]}).")
                _assert(W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]}).")
            # elif not self.dynamic_img_pad:
            #     _assert(
            #         H % self.patch_size[0] == 0,
            #         f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            #     )
            #     _assert(
            #         W % self.patch_size[1] == 0,
            #         f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
            #     )
        # if self.dynamic_img_pad:
        #     pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        #     pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        #     x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCDHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x




if __name__ == '__main__':
    model = vit_base_patch16()
    # print(model)
    x = torch.randn(1, 50, 96, 96)
    y = model(x)
    print(y.shape)