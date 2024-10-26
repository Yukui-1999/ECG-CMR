from utils.registry_class import ECGCLIP

from functools import partial
from torchinfo import summary
import torch
import torch.nn as nn
import timm.models.vision_transformer

@ECGCLIP.register_class()
class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,condition_dim=24,pretrained=None, **kwargs):
        super(ECGEncoder, self).__init__(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),img_size=(12,5000),patch_size=(1,100),in_chans=1,**kwargs)

        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], batch_first=True)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        
        # self.fcs = nn.Sequential(
        #         nn.Linear(condition_dim, 1024 * 2),
        #         nn.ReLU()
        #     )
        if pretrained is not None:
            self.init_from_ckpt(pretrained, ignore_keys=['head.weight', 'head.bias'])
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        #找到以ECG_encoder开头的key，组成新的sd，然后load_state_dict
        new_sd = {}
        for k in sd.keys():
            if k.startswith("ECG_encoder"):
                new_sd[k[12:]] = sd[k]
        sd = new_sd
        # print(sd.keys())
        # exit()
        keys = list(sd.keys())
        # print('this is the keys', keys)
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(msg)
    def forward_features(self, x, localized=False):
        B = x.shape[0]
        x = self.patch_embed(x)      
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        # if cond is not None:
        #     modulation_params = self.fcs(cond)
        #     gamma, beta = modulation_params.chunk(2, dim=-1)
        #     gamma = gamma.unsqueeze(1)
        #     beta = beta.unsqueeze(1)
        #     x = gamma * x + beta

        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)

if __name__ == '__main__':
    model = ECGEncoder(pretrained='/mnt/data2/dingzhengyao/work/checkpoint/preject_version1/output_dir/10002/checkpoint-45-loss-1.53.pth')
    summary(model, input_size=(16,1, 12, 5000))
    # out = model(torch.randn(1, 1, 12, 5000))
    # print(out.shape)