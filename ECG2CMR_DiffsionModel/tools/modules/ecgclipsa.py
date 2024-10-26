from utils.registry_class import ECGCLIPsa

from functools import partial
from torchinfo import summary
import torch
import torch.nn as nn
import timm.models.vision_transformer

@ECGCLIPsa.register_class()
class ECGCLIPsa(nn.Module):
    def __init__(self, pretrained = None, pretrained_both = None, **kwargs):
        super(ECGCLIPsa, self).__init__()
        self.encodersa = ECGEncoder()
        self.encdoerboth = ECGEncoder()
        if pretrained:
            ecg_checkpoint = torch.load(pretrained, map_location='cpu')
            ecg_checkpoint_model = ecg_checkpoint['model']
            ecg_checkpoint_model = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ECG_encoder')}
            #remove the prefix 'ECG_encoder.'
            ecg_checkpoint_model = {k.replace('ECG_encoder.', ''): v for k, v in ecg_checkpoint_model.items()}
            #remove keys startswith 'head'
            ecg_checkpoint_model = {k: v for k, v in ecg_checkpoint_model.items() if not k.startswith('head')}
            msg = self.encodersa.load_state_dict(ecg_checkpoint_model, strict=False)
            print(f'sa load from {pretrained} : {msg}')
        if pretrained_both:
            ecg_checkpoint = torch.load(pretrained_both, map_location='cpu')
            ecg_checkpoint_model = ecg_checkpoint['model']
            ecg_checkpoint_model = {k: v for k, v in ecg_checkpoint_model.items() if k.startswith('ECG_encoder')}
            #remove the prefix 'ECG_encoder.'
            ecg_checkpoint_model = {k.replace('ECG_encoder.', ''): v for k, v in ecg_checkpoint_model.items()}
            #remove keys startswith 'head'
            ecg_checkpoint_model = {k: v for k, v in ecg_checkpoint_model.items() if not k.startswith('head')}
            msg = self.encdoerboth.load_state_dict(ecg_checkpoint_model, strict=False)
            print(f'both load from {pretrained_both} : {msg}')
    def forward(self, x, cond):
        feature1 = self.encodersa(x, cond)
        feature2 = self.encdoerboth(x, cond)
        feature = torch.cat([feature1, feature2], dim=1)
        # print(f'feature shape: {feature.shape}')
        return feature

        

        


class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,condition_dim=24, **kwargs):
        super(ECGEncoder, self).__init__(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),img_size=(12,5000),patch_size=(1,100),in_chans=1,**kwargs)

        self.fcs = nn.Sequential(
                nn.Linear(condition_dim, 1024 * 2),
                nn.ReLU()
            )
        
    def forward_features(self, x,cond):
        B = x.shape[0]
        x = self.patch_embed(x)      
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if cond is not None:
            modulation_params = self.fcs(cond)
            gamma, beta = modulation_params.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = gamma * x + beta

        x = self.norm(x)
        return x

    def forward(self, x, cond):
        return self.forward_features(x, cond)

if __name__ == '__main__':
    model = ECGCLIPsa(pretrained="/data1/dingzhengyao/PretrainedECGclip/checkpoint-83-loss-0.51_sa.pth",
                      pretrained_both="/data1/dingzhengyao/PretrainedECGclip/checkpoint-50-loss-0.89_both.pth")
    input = torch.randn(1, 1, 12, 5000)
    cond = torch.randn(1, 24)
    out = model(input, cond)
    print(out.shape)
    # out = model(torch.randn(1, 1, 12, 5000))
    # print(out.shape)