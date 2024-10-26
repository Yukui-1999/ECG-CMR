# Created by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed,get_3d_sincos_pos_embed


class BaseGreenModel(nn.Module):
    
    def apply_mask(self, x, mask, patches_resolution):
        # mask out some patches according to the random mask
        B, N, C = x.shape
        H, W = patches_resolution
        print(f'B:{B},N:{N},C:{C},H:{H},W:{W}')
        mask = mask[:1].clone() # we use the same mask for the whole batch
        print(f'mask shape:{mask.shape}')
        print(mask)
        up_ratio = N // mask.shape[1]
        assert up_ratio * mask.shape[1] == N
        num_repeats = int(up_ratio**0.5)
        if up_ratio > 1:   # mask_size != patch_embed_size
            Mh, Mw = [sz // num_repeats for sz in patches_resolution]
            mask = mask.reshape(1, Mh, 1, Mw, 1)
            mask = mask.expand(-1, -1, num_repeats, -1, num_repeats)
            mask = mask.reshape(1, -1)
        
        # record the corresponding coordinates of visible patches
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1)  # H W 2
        coords = coords.reshape(1, H*W, 2)

        # mask out patches
        vis_mask = ~mask    # ~mask means visible, (1, N_vis)
        print(f'vis_mask shape:{vis_mask.shape},x shape:{x.shape}')
        x_vis = x[vis_mask.expand(B, -1)].reshape(B, -1, C)
        coords = coords[vis_mask].reshape(1, -1, 2) # (1 N_vis 2)

        return x_vis, coords, vis_mask

    def patchify(self, x):
        raise NotImplementedError()
    
    def forward_features(self, x, mask):
        raise NotImplementedError()
    
    def forward(self, x, mask):
        z_vis = self.forward_features(x, mask)
        return z_vis


class MaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, encoder, embed_dim, patch_size, in_chans=3,
                 decoder_num_patches=196, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 block_cls=Block, mlp_ratio=4,
                 **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.encoder = encoder
        self.num_patches = decoder_num_patches
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.final_patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if hasattr(self.encoder, 'patch_embed'):
            w = self.encoder.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # 
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size or self.final_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size or self.final_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        NOTE: Perform PER-BATCH random masking by per-sample shuffling.
        Per-batch shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = 1, self.num_patches  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        print(f'noise shape:{noise.shape}')
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        print(f'ids_keep shape:{ids_keep.shape}')
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        print(f'mask shape:{mask.shape}')
        mask.scatter_add_(1, ids_keep, torch.full([N, len_keep], fill_value=-1, dtype=mask.dtype, device=x.device))
        assert (mask.gather(1, ids_shuffle).gather(1, ids_restore) == mask).all()
        print(f'mask shape:{mask.shape}')
        # repeat the mask
        ids_restore = ids_restore.repeat(x.shape[0], 1)
        mask = mask.repeat(x.shape[0], 1)
        print(f'mask shape:{mask.shape}')
        return mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # generate random mask
        mask, ids_restore = self.random_masking(x, mask_ratio)
        print(f'mask shape:{mask.shape},ids_restore shape:{ids_restore.shape}')
        # L -> L_vis
        latent = self.encoder(x, mask.bool())

        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        print(f'x shape:{x.shape}')
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x_ + self.decoder_pos_embed
        print(f'x shape:{x.shape}')
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        print(f'x shape:{x.shape}')
        # predictor projection
        x = self.decoder_pred(x)
        print(f'x shape:{x.shape}')
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        print(f'latent shape:{latent.shape},mask shape:{mask.shape},ids_restore shape:{ids_restore.shape}')
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        print(f'pred shape:{pred.shape}')
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, latent




# --------------------------------------------------------
# 3D base green model
class BaseGreenModel3D(nn.Module):
    def __init__(self):

        super(BaseGreenModel3D, self).__init__()
        # Initialize layers and parameters here

    def apply_mask(self, x, depth, height, width, mask, patches_resolution):

        # mask out some patches according to the random mask
        B, N, C = x.shape
        D, H, W = patches_resolution  # N = D*H*W

        # mask (D // 8) * (H // 8) * (W // 8)
        mask = mask[:1].clone()  # we use the same mask for the whole batch

        up_ratio = N // mask.shape[1]
        assert up_ratio * mask.shape[1] == N
        num_repeats = int(round(up_ratio ** (1 / 3)))  #
        if up_ratio > 1:  # mask_size != patch_embed_size
            Md, Mh, Mw = [sz // num_repeats for sz in patches_resolution]
            mask = mask.reshape(1, Md, 1, Mh, 1, Mw, 1)
            mask = mask.expand(-1, -1, num_repeats, -1, num_repeats, -1, num_repeats)
            mask = mask.reshape(1, -1)

        # record the corresponding coordinates of visible patches
        coords_d = torch.arange(D, device=x.device)
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]), dim=-1)  # D H W 3
        coords = coords.reshape(1, D*H*W, 3)

        # mask out patches
        vis_mask = ~mask  # ~mask means visible, (1, N_vis)
        x_vis = x[vis_mask.expand(B, -1)].reshape(B, -1, C)
        coords = coords[vis_mask].reshape(1, -1, 3)  # (1 N_vis 3)

        return x_vis, coords, vis_mask

    def patchify(self, x):
        raise NotImplementedError()

    def forward_features(self, x, mask):
        # Implement the feature extraction process
        raise NotImplementedError()

    def forward(self, x, mask):
        z_vis = self.forward_features(x, mask)
        return z_vis




class MaskedAutoencoder3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for 3D data """
    def __init__(self, encoder, embed_dim, patch_size=32, in_chans=1, grid_size=(2, 7, 7),
                 decoder_num_patches=196, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 block_cls=Block, mlp_ratio=4,
                 **kwargs):
        super().__init__()

        # MAE encoder specifics
        self.encoder = encoder
        self.grid_size = grid_size
        self.num_patches = decoder_num_patches
        assert decoder_num_patches == grid_size[0] * grid_size[1] * grid_size[2], \
            f'decoder_num_patches {decoder_num_patches} != grid_size {grid_size} product.'

        # MAE decoder specifics
        self.final_patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            block_cls(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        pd, ph, pw = patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, pd * ph * pw * in_chans, bias=True) # encoder to decoder

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # grid size: change this with patch size
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size,
                                                    cls_token=False)

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        if hasattr(self.encoder, 'patch_embed'):
            w = self.encoder.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, 1, D, H, W)
        x: (N, L, patch_size**3 *1)
        """
        p = patch_size or self.final_patch_size
        assert imgs.shape[2] % p[0] == 0 and imgs.shape[3] % p[1] == 0 and imgs.shape[4] % p[2] == 0

        d = imgs.shape[2] // p[0]
        h = imgs.shape[3] // p[1]
        w = imgs.shape[4] // p[2]
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p[0], h, p[1], w, p[2]))  # x: (N, 1, d, p1, h, p2, w, p3)
        # rearrange to (N, d*h*w, p1*p2*p3*1)
        x = rearrange(x, 'n c d p1 h p2 w p3 -> n (d h w) (p1 p2 p3 c)')
        # x = x.reshape(imgs.shape[0], d * h * w, p[0] * p[1] * p[2] * 1)
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, D, H, W)
        """
        p = patch_size or self.final_patch_size
        d = h = w = int(x.shape[1]**(1/3))  # FIXME: check this

        assert d * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, 1))
        x = torch.einsum('ndhpwqc->ncdphwpq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        NOTE: Perform PER-BATCH random masking by per-sample shuffling.
        Per-batch shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = 1, self.num_patches  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_add_(1, ids_keep, torch.full([N, len_keep], fill_value=-1, dtype=mask.dtype, device=x.device))
        assert (mask.gather(1, ids_shuffle).gather(1, ids_restore) == mask).all()

        # repeat the mask
        ids_restore = ids_restore.repeat(x.shape[0], 1)
        mask = mask.repeat(x.shape[0], 1)

        return mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # generate random mask
        mask, ids_restore = self.random_masking(x, mask_ratio)

        # L -> L_vis
        latent = self.encoder(x, mask.bool())

        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, D, H, W]
        pred: [N, L, pd*ph*pw*C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        # normalize pixel loss,
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, latent
