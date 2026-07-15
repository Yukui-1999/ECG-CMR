from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from CMR_generation.CMR_gen_models.diffloss import DiffLoss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(50, 96, 96), vae_stride=(2, 8, 8), patch_size=(5, 1, 1),
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 use_mae_loss=False, # 是否对z使用mae损失
                 coef_mae_loss=1.0,
                 use_rep_cond=False,
                 diffloss_on_rep=True,
                 ecg_model=None,
                 ecg_latent_dim=None,
                 ):
        super().__init__()

        self.use_rep_cond = use_rep_cond
        self.diffloss_on_rep = diffloss_on_rep
        print(f"MAR model with rep_cond: {self.use_rep_cond}, diffloss_on_rep: {self.diffloss_on_rep}")
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size # 原始图像大小
        self.vae_stride = vae_stride # vae下采样倍数
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size[-1] // vae_stride[-1] // patch_size[-1]
        self.seq_t = img_size[0] // vae_stride[0] // patch_size[0]
        self.seq_len = self.seq_h * self.seq_w * self.seq_t
        self.token_embed_dim = vae_embed_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        # self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        # self.ecg_emb = nn.Linear(768, encoder_embed_dim) # align的输入要从768改成320
        # self.ecg_emb = nn.Sequential(
        #     nn.Linear(768, encoder_embed_dim),
        #     nn.Linear(encoder_embed_dim, encoder_embed_dim)
        # )
        if ecg_model == 'base':
            self.ecg_emb = nn.Sequential(
                nn.Linear(768, encoder_embed_dim),
                nn.Linear(encoder_embed_dim, encoder_embed_dim)
            )
        elif ecg_model == 'small':
            self.ecg_emb = nn.Sequential(
                nn.Linear(512, encoder_embed_dim),
                nn.Linear(encoder_embed_dim, encoder_embed_dim)
            )
        elif ecg_model == 'stmem':
            self.ecg_emb = nn.Sequential(
                nn.Linear(ecg_latent_dim if ecg_latent_dim is not None else 768, encoder_embed_dim),
                nn.Linear(encoder_embed_dim, encoder_embed_dim)
            )
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.fake_rep_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        self.rep_embedder = nn.Linear(encoder_embed_dim, encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * patch_size[2] * vae_embed_dim, bias=True)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self.use_mae_loss = use_mae_loss
        self.coef_mae_loss = coef_mae_loss

    def initialize_weights(self):
        # parameters
        # torch.nn.init.normal_(self.ecg_emb.weight, std=.02)
        torch.nn.init.normal_(self.ecg_emb[0].weight, std=.02)
        torch.nn.init.normal_(self.ecg_emb[1].weight, std=.02)

        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # 初始化会影响maeloss的速度
        torch.nn.init.normal_(self.fake_rep_latent, std=.02)
        torch.nn.init.normal_(self.rep_embedder.weight, std=.02)
        torch.nn.init.normal_(self.decoder_pred.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, t, h, w = x.shape  # batch_size, channels, time_steps, height, width
        p = self.patch_size  # patch size for both spatial and temporal chunks
        
        # 计算每个维度的块数
        t_ = t // p[0]  # 分块后的时间维度大小
        h_ = h // p[1]  # 分块后的空间高度
        w_ = w // p[2]  # 分块后的空间宽度

        # 重塑形状：将每个维度分成多个块
        x = x.reshape(bsz, c, t_, p[0], h_, p[1], w_, p[2])
        
        # 使用einsum进行重新排列
        x = torch.einsum('nctphqwr->nthwcpqr', x)
        
        # 将结果重新整理成合适的形状
        x = x.reshape(bsz, t_ * h_ * w_, c * p[0] * p[1] * p[2])
        
        return x # bs, seqlen, patch_dim
    
    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        t_, h_, w_ = self.seq_t, self.seq_h, self.seq_w

        x = x.reshape(bsz, t_, h_, w_, c, p[0], p[1], p[2])
        x = torch.einsum('nthwcpqr->nctphqwr', x)
        x = x.reshape(bsz, c, t_ * p[0], h_ * p[1], w_ * p[2])
        return x # [n, c, t, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        # x = x[torch.randperm(bsz * seq_len)].view(bsz, seq_len, embed_dim)
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1) # buffersize的内容全部替换成class_embedding

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping masked tokens
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        if self.training:
            return x, drop_latent_mask
        return x, None

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]

        x_diff = x + self.diffusion_pos_embed_learned
        return x, x_diff

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class(metrics) embed
        class_embedding = self.ecg_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x, drop_latent_mask = self.forward_mae_encoder(x, mask, class_embedding)
        # mae decoder
        z, z_diff = self.forward_mae_decoder(x, mask) # (bs, seq_len, embed_dim)

        if self.training and self.use_rep_cond:
            rep = x[:, :self.buffer_size].mean(dim=1) # global pool (bs, embed_dim) # 全部mask的时候也要有特征，所以要把buffer也放进来
            # print(rep)
            if self.diffloss_on_rep is False:
                rep_diff = rep.clone().detach() # diffloss不优化rep
            else: rep_diff = rep

            rep_diff = drop_latent_mask * self.fake_rep_latent + (1 - drop_latent_mask) * rep_diff
            rep_diff = self.rep_embedder(rep_diff) # (bs, embed_dim)
            z_diff = z_diff + rep_diff.unsqueeze(1) # rep广播到相同大小

        # if self.training and self.use_rep_cond:
        #     rep = x[:, :self.buffer_size].mean(dim=1) # global pool (bs, embed_dim) # 全部mask的时候也要有特征，所以要把buffer也放进来

        #     rep = drop_latent_mask * self.fake_rep_latent + (1 - drop_latent_mask) * rep
        #     rep = self.rep_embedder(rep) # (bs, embed_dim)
        #     if self.diffloss_on_rep is False:
        #         rep_diff = rep.clone().detach() # diffloss不优化rep
        #     else: rep_diff = rep
        #     z_diff = z_diff + rep_diff.unsqueeze(1) # rep广播到相同大小

        z = self.decoder_pred(z)
        if self.use_mae_loss:
            mae_loss = (z - gt_latents) ** 2
            mae_loss = mae_loss.mean(dim=-1)  # [N, L], mean loss per patch
            mae_loss = (mae_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(z)

        # diffloss
        loss = self.forward_loss(z=z_diff, target=gt_latents, mask=mask)

        loss_dict = {'diffloss': loss.detach().cpu().item(), 'mae_loss': mae_loss.detach().cpu().item()} if self.use_mae_loss else {'diffloss': loss.detach().cpu().item(), 'mae_loss': 0}
        loss = loss + self.coef_mae_loss * mae_loss if self.use_mae_loss else loss
        return loss, loss_dict

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.ecg_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x, _ = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            _, z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
