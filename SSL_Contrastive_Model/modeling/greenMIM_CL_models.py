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

import torch
import torch.nn.functional as F
from modeling.base_green_models import MaskedAutoencoder
from modeling.model_factory import green_mim_swin_base_patch4_dec512b1,green_mim_swin_base_patch4_win14_dec512b1,green_mim_swin_large_patch4_win14_dec512b1
class ClipLoss(torch.nn.Module):

    def __init__(self, temperature,args):
        super(ClipLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = temperature
        self.device = args.device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        
    # def softXEnt(self, target, logits):
    #     logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
    #     loss = -(target * logprobs).sum() / logits.shape[0]
    #     return loss

    def forward(self, zis, zjs,norm=True):
        temperature = self.temperature
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        # labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = torch.arange(len(hidden1)).to(self.device)
        logits = torch.matmul(hidden1, torch.transpose(hidden2,0, 1)) / temperature
        zis_findmostgood_zjs = F.cross_entropy(logits, labels)
        zjs_findmostgood_zis = F.cross_entropy(torch.transpose(logits,0, 1), labels)
        # loss = F.cross_entropy(logits, labels) + F.cross_entropy(torch.transpose(logits,0, 1), labels)
        # print(f'zis_findmostgood_zjs:{zis_findmostgood_zjs},zjs_findmostgood_zis:{zjs_findmostgood_zis}')
        loss = 0.5 * zis_findmostgood_zjs + 0.5 * zjs_findmostgood_zis
        return loss



class CLMaskedAutoencoder(nn.Module):
    def __init__(self,mask_model1,mask_model2,args):
        super(CLMaskedAutoencoder, self).__init__()
        self.mask_model1 = mask_model1
        self.mask_model2 = mask_model2
        self.linear1 = nn.Linear(args.embedding_dim,256)
        self.linear2 = nn.Linear(args.embedding_dim,256)
        self.clloss =  ClipLoss(0.1,args=args)
        self.fc_norm1 = nn.LayerNorm(args.embedding_dim)
        self.fc_norm2 = nn.LayerNorm(args.embedding_dim)


    def forward(self,imgs1,imgs2,mask_ratio=0.75):
        loss1, pred1, mask1, latent1 = self.mask_model1(imgs1, mask_ratio)
        loss2, pred2, mask2, latent2 = self.mask_model2(imgs2, mask_ratio)
        # print(f'latent1 shape:{latent1.shape},latent2 shape:{latent2.shape}')

        latent1 = latent1[:, :, :].mean(dim=1) 
        latent1 = self.fc_norm1(latent1)
        latent2 = latent2[:, :, :].mean(dim=1)
        latent2 = self.fc_norm2(latent2)

        feature1 = self.linear1(latent1)
        feature2 = self.linear2(latent2)
        cliploss = self.clloss(feature1,feature2)
        total_loss = loss1 + loss2 + cliploss
        loss_dict = {'loss1':loss1,'loss2':loss2,'cliploss':cliploss}
        return loss_dict,latent1,latent2,feature1,feature2

if __name__ == '__main__':
    
    print(torch.__version__)
    print(torch.cuda.is_available())

    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = {
        'batch_size':2,
        'device':'cuda:0'
    }
    args = Args(**args)
    model1 = green_mim_swin_base_patch4_dec512b1().to(args.device)
    model2 = green_mim_swin_base_patch4_dec512b1().to(args.device)
    clmodel = CLMaskedAutoencoder(model1,model2,args).to(args.device)
    imgs1 = torch.randn(2,50,224,224).to(args.device)
    imgs2 = torch.randn(2,50,224,224).to(args.device)
    loss = clmodel(imgs1,imgs2)
    print(loss)

