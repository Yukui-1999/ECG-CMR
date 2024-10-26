import torch
import torch.nn as nn

class ECG_finetune_Bimodel(nn.Module):
    def __init__(self, model1, model2, embed_dim, args):
        super(ECG_finetune_model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, args.latent_dim)

    def forward(self, x, cond):
        features1 = self.model1.forward_features(x, cond)
        features2 = self.model2.forward_features(x, cond)
        x = self.bilinear(features1, features2)
        x = self.head(x)
        return x
