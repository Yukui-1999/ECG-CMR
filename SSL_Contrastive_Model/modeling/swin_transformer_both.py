# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import torch
import torch.nn as nn
from modeling.swin_transformer_other import build_swin
from modeling.swin_transformer import build_swin as build_swin_woCond
from util.extract_backbone import load_pretrained_

class SwinTransformerBoth(nn.Module):
    def __init__(self, config,args):
        super(SwinTransformerBoth, self).__init__()
        if args.with_cond:
            self.swin1 = build_swin(config)
            self.swin2 = build_swin(config)
        else:
            self.swin1 = build_swin_woCond(config)
            self.swin2 = build_swin_woCond(config)
        self.fc = nn.Linear(1024 * 2, config.MODEL.NUM_CLASSES)
        if args.pretrained_swin:
            print("load pretrained swin cmr_model")
            cmr_checkpoint = torch.load(args.pretrained_swin, map_location='cpu')
            cmr_checkpoint_model = cmr_checkpoint['model']
            cmr_checkpoint_model1 = {k: v for k, v in cmr_checkpoint_model.items() if k.startswith('mask_model1')}
            # replace mask_model1 with ''
            cmr_checkpoint_model1 = {k.replace('mask_model1.', ''): v for k, v in cmr_checkpoint_model1.items()}
            cmr_checkpoint_model2 = {k: v for k, v in cmr_checkpoint_model.items() if k.startswith('mask_model2')}
            # replace mask_model2 with ''
            cmr_checkpoint_model2 = {k.replace('mask_model2.', ''): v for k, v in cmr_checkpoint_model2.items()}

            load_pretrained_(ckpt_model=cmr_checkpoint_model1, model=self.swin1)
            load_pretrained_(ckpt_model=cmr_checkpoint_model2, model=self.swin2)
    def forward(self, x1, x2, cond):
        feat1 = self.swin1.forward_features(x1, cond)
        feat2 = self.swin2.forward_features(x2, cond)
        feat = torch.cat([feat1, feat2], dim=1)
        logits = self.fc(feat)
        return logits

