import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.jit import Final
from typing import Callable, List, Optional, Tuple, Union
from timm.layers import PatchEmbed, Mlp, DropPath, use_fused_attn
import torch.nn.functional as F

from models import encoder
from CMR_encoder import models_vit


def build_ecg_model(ecg_config_path):
    
    with open(os.path.realpath(ecg_config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = 1
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] == "pretrain":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
    elif config['mode'] == "align":
        checkpoint = torch.load(config['afterAlign_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['afterAlign_path']}")
    elif config['mode'] == "scratch":
        print('Training from scratch')
        return ecg_model
    else:
        raise ValueError(f'Unsupported mode: {config["mode"]}')
    
    # load pre-trained weights
    checkpoint_model = checkpoint['model']
    state_dict = ecg_model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Remove key {k} from pre-trained checkpoint")
            del checkpoint_model[k]
    msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
    print(f'Load pre-trained ECG model: {msg}')


    return ecg_model

 
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ECG_genCMR_model(torch.nn.Module):
    def __init__(self, args, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_norm=False, init_values=None, proj_drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 mlp_layer=Mlp, use_fc_norm=False,pred_metric=False,by_block=True):

        super(ECG_genCMR_model, self).__init__()
        self.by_block = by_block
        self.ecg_model = build_ecg_model(args.ecg_config_path)
        self.cmr_model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
        )
        checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')

        checkpoint_model = checkpoint['model']
        msg = self.cmr_model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained CMR model: {msg}')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.GELU(),
        )
        if pred_metric:
            print(f'Using prediction metric head')
            self.head = nn.Sequential(
                    nn.Linear(embed_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 82)
                )
        else:
            self.head = nn.Linear(embed_dim, args.num_classes)
            
            
    def forward(self, ecg, cmr):
        _ = self.ecg_model(ecg)
        ecg_latent = self.ecg_model.latent # torch.Size([b, 12, 30, 768])
        _ = self.cmr_model(cmr) # torch.Size([b, 37, 768])
        cmr_latent = self.cmr_model.latent
        
        if self.by_block:
            ecg_latent = torch.mean(ecg_latent, dim=1) # torch.Size([b, 30, 768])
            ecg_cmr_latent = torch.cat((ecg_latent, cmr_latent), dim=1) # torch.Size([b, 30+37, 768])
            ecg_cmr_latent = self.blocks(ecg_cmr_latent)
            ecg_cmr_latent = torch.mean(ecg_cmr_latent, dim=1)
            out = self.norm(ecg_cmr_latent)
            out = self.head(out)
            return out
        else:
            ecg_latent = torch.mean(ecg_latent, dim=(1, 2))
            cmr_latent = cmr_latent[:,0,:]
            assert ecg_latent.dim() == 2
            assert cmr_latent.dim() == 2
            ecg_cmr_latent = torch.cat((ecg_latent, cmr_latent), dim=1) # torch.Size([b, 2*768])
            ecg_cmr_latent = self.gate(ecg_cmr_latent)
            out = self.norm(ecg_cmr_latent)
            out = self.head(out)
            return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecg_config_path', type=str, default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/Cla/st_mem_align.yaml')
    parser.add_argument('--cmr_model', type=str, default='vit_base_patch16')
    parser.add_argument('--cmr_pretrained_weights', type=str, default='/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth')
    parser.add_argument('--drop_path', type=float, default=0)
    parser.add_argument('--num_classes', type=int, default=1)
    args = parser.parse_args()

    model = ECG_genCMR_model(args,pred_metric=True)
    ecg = torch.randn(1, 12, 2250)
    cmr = torch.randn(1, 50, 96, 96)
    out = model(ecg, cmr)
    print(out.shape)