# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import forward
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

class ClipLoss(torch.nn.Module):

    def __init__(self, temperature,args):
        super(ClipLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = temperature
        self.device = args.device
        

    def forward(self, zis, zjs,norm=True):
        temperature = self.temperature
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
        hidden1, hidden2 = zis, zjs
        labels = torch.arange(len(hidden1)).to(self.device)
        logits = torch.matmul(hidden1, torch.transpose(hidden2,0, 1)) / temperature
        zis_findmostgood_zjs = F.cross_entropy(logits, labels)
        zjs_findmostgood_zis = F.cross_entropy(torch.transpose(logits,0, 1), labels)
        loss = 0.5 * zis_findmostgood_zjs + 0.5 * zjs_findmostgood_zis
        return loss

def build_loss_fn(config: dict) -> Tuple[nn.Module, nn.Module]:
    loss_name = config['name']
    if loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
        output_act = nn.Softmax(dim=-1)
    elif loss_name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        output_act = nn.Sigmoid()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
    return loss_fn, output_act





def cosine_similarity_matrix(A, B):
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    return A @ B.t()

def pairwise_sq_l2(A, B):
    a2 = (A**2).sum(dim=1, keepdim=True)
    b2 = (B**2).sum(dim=1, keepdim=True).t()
    ab = A @ B.t()
    return a2 - 2*ab + b2

# -----------------------------
# Losses
# -----------------------------
def phenotype_weight_matrix(phi: torch.Tensor, sigma: float) -> torch.Tensor:
    """ W_ik = 1 - exp(-||phi_i - phi_k||^2 / sigma^2) ; diag=0 """
    dist2 = pairwise_sq_l2(phi, phi)  # [B,B]
    W = 1.0 - torch.exp(-dist2 / (sigma**2 + 1e-12))
    W.fill_diagonal_(0.0)
    return W

def weighted_infonce(z_q, z_k, W_neg, tau=0.07):
    """ 对称调用时：两次互换 z_q, z_k """
    sim = cosine_similarity_matrix(z_q, z_k) / tau  # [B,B]
    exp_sim = torch.exp(sim)
    # 正样本对角线保留；负样本乘权重
    weighted_exp = exp_sim * W_neg + torch.diag(exp_sim.diag())
    denom = weighted_exp.sum(dim=1) + 1e-12
    pos = sim.diag().exp()
    loss = -torch.log(pos / denom + 1e-12).mean()
    return loss


def supcon_softpositives(z_q, z_k, phi, eps, tau=0.07, knn_fallback=10):
    B = z_q.size(0)
    sim = cosine_similarity_matrix(z_q, z_k) / tau
    dist = torch.sqrt(pairwise_sq_l2(phi, phi) + 1e-12)

    device = sim.device
    I = torch.eye(B, device=device, dtype=torch.float32)

    # 先按 eps 得到正集合
    pos_mask = (dist <= eps).float()
    pos_mask = pos_mask * (1 - I)  # 去掉对角

    # 兜底：对每个 i，如果没有正样本，用表型距离最近的 k 个作为正样本
    no_pos = (pos_mask.sum(dim=1) == 0)  # [B]
    if no_pos.any():
        # 距离越小越相似；取最小的 k+1（包含自身），再去掉自身
        k = min(knn_fallback, B-1)
        dist_clone = dist.clone()
        # 先把对角设大，避免选到自己
        dist_clone.fill_diagonal_(float('inf'))
        knn_idx = dist_clone.topk(k=k, largest=False).indices  # [B,k]
        # 构建补充的正样本 mask
        fallback_mask = torch.zeros_like(pos_mask)
        rows = no_pos.nonzero(as_tuple=True)[0]
        fallback_mask[rows.unsqueeze(1), knn_idx[rows]] = 1.0
        pos_mask = torch.maximum(pos_mask, fallback_mask)

    exp_sim = torch.exp(sim)
    denom = (exp_sim * (1 - I)).sum(dim=1, keepdim=True) + 1e-12

    # 平均正样本 log-prob
    log_prob = torch.log((exp_sim / denom) + 1e-12)
    pos_counts = pos_mask.sum(dim=1)  # 至少 >=1
    pos_logprob_sum = (log_prob * pos_mask).sum(dim=1)
    li = - pos_logprob_sum / (pos_counts + 1e-12)
    return li.mean()
# def supcon_softpositives(z_q, z_k, phi, eps, tau=0.07):
#     """ 软正样本 SupCon：表型距离<=eps 作为正集合 """
#     B = z_q.size(0)
#     sim = cosine_similarity_matrix(z_q, z_k) / tau  # [B,B]
#     dist = torch.sqrt(pairwise_sq_l2(phi, phi) + 1e-12)
#     pos_mask = (dist <= eps).float()
#     pos_mask.fill_diagonal_(1.0)
#     exp_sim = torch.exp(sim)
#     denom = (exp_sim * (1 - torch.eye(B, device=sim.device))).sum(dim=1, keepdim=True) + 1e-12
#     pos_only = pos_mask * (1 - torch.eye(B, device=sim.device))
#     log_prob = torch.log((exp_sim / denom) + 1e-12)
#     pos_counts = pos_only.sum(dim=1)
#     pos_logprob_sum = (log_prob * pos_only).sum(dim=1)
#     li = - pos_logprob_sum / (pos_counts + 1e-12)
#     return li.mean()


from typing import Dict, Tuple, Optional, List
class weighted_infonce_loss(torch.nn.Module):
    def __init__(self, args=None):
        super(weighted_infonce_loss, self).__init__()
        self.cfg = args
    
    def forward(self, ze, zc, phi) -> Tuple[torch.Tensor, Dict[str,float]]:
        cfg = self.cfg
        W = phenotype_weight_matrix(phi, cfg.sigma)
        loss_e2c = weighted_infonce(ze, zc, W, tau=cfg.tau)
        loss_c2e = weighted_infonce(zc, ze, W, tau=cfg.tau)
        loss_main = 0.5*(loss_e2c + loss_c2e)
        metrics = {"loss_main": float(loss_main.item())}

        if cfg.use_supcon:
            ls = 0.5*(supcon_softpositives(ze, zc, phi, cfg.eps, cfg.tau) +
                      supcon_softpositives(zc, ze, phi, cfg.eps, cfg.tau))
            loss_main = loss_main + ls
            metrics["loss_supcon"] = float(ls.item())

        
        loss_total = loss_main

        metrics["loss_total"] = float(loss_total.item())
        return loss_total, metrics
    

def barlow_twins_loss(z1, z2, lambd=5e-3):
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-12)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-12)
    C = (z1.t() @ z2) / B
    on_diag  = torch.diagonal(C).add_(-1).pow_(2).sum()
    off_diag = (C - torch.eye(D, device=C.device)).pow(2).sum() - on_diag
    return on_diag + lambd * off_diag

def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    inv = F.mse_loss(z1, z2)
    def variance_term(z):
        std = z.std(dim=0)
        return torch.mean(F.relu(eps - std))
    var = variance_term(z1) + variance_term(z2)
    def covariance_term(z):
        z = z - z.mean(dim=0)
        B, D = z.size()
        c = (z.t() @ z) / (B - 1)
        off_diag = c - torch.diag(torch.diag(c))
        return (off_diag.pow(2).sum()) / D
    cov = covariance_term(z1) + covariance_term(z2)
    return sim_coeff*inv + var_coeff*var + cov_coeff*cov
