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
from visualizer import get_local
import torch
import torch.nn as nn
import timm.models.vision_transformer


class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,args=None,condition_dim=24, **kwargs):
        super(ECGEncoder, self).__init__(**kwargs)
        
        condition_dim = condition_dim
        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], batch_first=True)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        self.fcs = nn.Sequential(
                nn.Linear(condition_dim, kwargs['embed_dim'] * 2),
                nn.ReLU()
            )
        
    def forward_features(self, x ,cond,localized=False):
        
        B = x.shape[0]
        x = self.patch_embed(x)      
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        # print(f'x.shape: {x.shape}')
        modulation_params = self.fcs(cond)
        gamma, beta = modulation_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # mean = x.mean(dim=1, keepdim=True)  # (b, 1, 1024)
        # std = x.std(dim=1, keepdim=True)  # (b, 1, 1024)
        # x_normalized = (x - mean) / (std + 1e-5)
        x = gamma * x + beta

        if localized:
            outcome = x[:, 1:]
        elif self.global_pool == "attention_pool":
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x,cond):
        features = self.forward_features(x,cond)
        x = features
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        
        return features,x


def vit_pluto_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=256, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=640, depth=6, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_big_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=768, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch200(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 200), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch100(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 100), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch50(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 50), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch10(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 10), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch224(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 224), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patchX_(**kwargs):
    model = ECGEncoder(
        embed_dim=1024, depth=16, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch112(**kwargs):
    model = ECGEncoder(
        patch_size=(65, 112), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    model = vit_large_patchX(img_size=(12,5000),patch_size=(1,100),in_chans=1,condition_dim=3)
    out = model(torch.randn(1, 1, 12, 5000),torch.randn(1, 3))
    print(out[1].shape)