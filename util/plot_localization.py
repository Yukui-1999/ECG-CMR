import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib import animation
import matplotlib.gridspec as gridspec
import seaborn as sns
import wandb


def plot_ecg_attention(original_ecg, attentation_map, idx, save_dir):
    """
    :input:
    original_ecg (B, C, C_sig, T_sig)
    attention_map (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    """
    # print(f'original_ecg shape: {original_ecg.shape}')
    B, C, C_sig, T_sig = original_ecg.shape
    B, Heads, N, N = attentation_map.shape

    NpC = int(N / C_sig) # N_(C_sig)

    # only for nice visualization 
    original_ecg = (original_ecg+0.5*abs(original_ecg.min()))

    # (B, Heads, N_(C_sig), N_(C_sig)), attention map of the first ecg lead
    # attentation_map = attentation_map[:, :, 1:(1+NpC), 1:(1+NpC)] # leave the cls token out
    # (B, Heads, N_(C_sig))
    attentation_map = attentation_map.mean(dim=2)
    attentation_map = F.normalize(attentation_map, dim=-1)
    attentation_map = attentation_map.softmax(dim=-1)
    # (B, Heads, T_sig)
    attentation_map = F.interpolate(attentation_map, size=T_sig, mode='linear')

    # (T_sig)
    original_ecg = original_ecg[idx, 0, 0].cpu()
    # (Heads, T_sig)
    attentation_map = attentation_map[idx].cpu()

    fig, axes = plt.subplots(nrows=Heads, sharex=True)

    for head in range(0, Heads):
        axes[head].plot(range(0, original_ecg.shape[-1], 1), original_ecg, zorder=2) # (2500)
        sns.heatmap(attentation_map[head, :].unsqueeze(dim=0).repeat(15, 1), linewidth=0.0, # (1, 2500)
                    alpha=0.3,
                    zorder=1,
                    ax=axes[head])
        axes[head].set_ylim(original_ecg.min(), original_ecg.max())

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()


    plt.savefig(f'{save_dir}/attn_ecg_{idx}.png')
    plt.close('all')

# def plot_image_localization(original_img, importance_img, idx, save_dir):
#     """
#     :input: 
#     original_img (B, C, H_img, H_img)
#     importance_img (B, H'_img, W'_img)
#     """
#     B, _, H_img, W_img = original_img.shape

#     original_img = original_img - original_img.min()
#     original_img = original_img / original_img.max()
#     original_img = original_img.cpu()

#     importance_img = F.interpolate(importance_img.unsqueeze(1), size=(H_img, W_img), mode='bilinear').squeeze(1)
#     importance_img = importance_img.cpu()
#     img_idx = int(torch.rand(1).item()*32)
#     original_img = original_img[idx][0].unsqueeze(dim=0)
#     importance_img = importance_img[idx]
#     # print(f'original_img shape: {original_img.shape}')
#     # print(f'importance_img shape: {importance_img.shape}')
#     # print(f'importance_img: {importance_img}')
#     importance_img = 2 * (importance_img - importance_img.min()) / (importance_img.max() - importance_img.min()) - 1
#     fig, ax = plt.subplots(figsize=(6, 6))  # 单子图正方形画布

#     # 先绘制原始灰度医学影像
#     ax.imshow(original_img.permute(1, 2, 0),  # 假设输入是 (C, H, W)
#             cmap='gray', 
#             aspect='equal',  # 关键！保持宽高比不变形
#             extent=[0, 96, 0, 96],  # 坐标范围对齐热力图
#             zorder=1)

#     # 叠加红蓝热力图（匹配图片中的颜色条）
#     sns.heatmap(importance_img,
#                 cmap="RdBu_r",
#                 vmin=-1.0, 
#                 vmax=1.0,
#                 alpha=0.4,  # 调整透明度让底层影像可见
#                 linewidth=0,
#                 square=True,  # 保持热力图正方形
#                 cbar_kws={
#                     "label": "",  # 隐藏颜色条标题
#                     "ticks": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
#                 },
#                 ax=ax,
#                 zorder=2)

#     # 取消坐标轴刻度
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/local_cmr_{idx}.png', dpi=300)
#     plt.close('all')

import os

def plot_image_localization(original_img, importance_img, idx, save_dir):
    """
    保存两个图像：
    1. 原图（grayscale）
    2. 原图 + 叠加热力图

    Parameters:
    - original_img: Tensor of shape (B, C, H, W)
    - importance_img: Tensor of shape (B, H', W')
    - idx: 要保存的图像索引
    - save_dir: 图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    B, _, H_img, W_img = original_img.shape

    # Normalize original image to [0, 1]
    original_img = original_img - original_img.min()
    original_img = original_img / original_img.max()
    original_img = original_img.cpu()

    # Resize importance map to match original image
    importance_img = F.interpolate(importance_img.unsqueeze(1), size=(H_img, W_img), mode='bilinear').squeeze(1)
    importance_img = importance_img.cpu()

    # 取第 idx 张
    original_img_single = original_img[idx][0].unsqueeze(0)  # shape (1, H, W)
    importance_img_single = importance_img[idx]

    # ---------- Step 1: 保存原图 ----------
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(original_img_single.squeeze(0), cmap='gray', aspect='equal', extent=[0, 96, 0, 96])
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/original_cmr_{idx}.png', dpi=300)
    plt.close(fig1)

    # ---------- Step 2: 保存热力图叠加图 ----------
    # Normalize importance to [-1, 1] for consistent coloring
    importance_img_single = 2 * (importance_img_single - importance_img_single.min()) / (importance_img_single.max() - importance_img_single.min()) - 1

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(original_img_single.permute(1, 2, 0), cmap='gray', aspect='equal', extent=[0, 96, 0, 96], zorder=1)
    sns.heatmap(importance_img_single,
                cmap="RdBu_r",
                vmin=-1.0, vmax=1.0,
                alpha=0.4,
                linewidth=0,
                square=True,
                cbar_kws={
                    "label": "",
                    "ticks": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
                },
                ax=ax2,
                zorder=2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/local_cmr_{idx}.png', dpi=300)
    plt.close(fig2)
    
    
def plot_ecg_localization(original_ecg, importance_ecg, idx, save_dir):
    """
    :input: 
    original_ecg (B, C, C_sig, T_sig)
    importance_ecg (B, C_sig, N'_(C_sig))
    """
    B, _, C_sig, T_sig = original_ecg.shape

    # only for nice visualization 
    original_ecg = (original_ecg+0.5*abs(original_ecg.min())).cpu()
    original_ecg = original_ecg.cpu()
    # (B, C_sig, T_sig)
    importance_ecg = F.interpolate(importance_ecg, size=T_sig, mode='linear')
    importance_ecg = importance_ecg.cpu()
    original_ecg = original_ecg[idx]
    importance_ecg = importance_ecg[idx]
    # print(f'in plot ecg local function')
    # print(f'original_ecg shape: {original_ecg.shape}')
    # print(f'importance_ecg shape: {importance_ecg.shape}')
    # print(f'importance_ecg: {importance_ecg}')
    importance_ecg = 2 * (importance_ecg - importance_ecg.min()) / (importance_ecg.max() - importance_ecg.min()) - 1

    fig, axes = plt.subplots(nrows=13, ncols=2, sharex=True, gridspec_kw={'width_ratios': [40, 1]}, figsize=(8, 12))
    
    for x in range(12):
        axes[x, 0].plot(range(0, original_ecg.shape[-1], 1), original_ecg[0, x, :], zorder=2, linewidth=0.6,color='black') # (2500)
        sns.heatmap(importance_ecg[x, :].unsqueeze(dim=0).repeat(15, 1), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 2500) cmap="RdBu_r"
                    alpha=0.2,
                    zorder=1,
                    ax=axes[x, 0])
        axes[x, 0].set_ylim(original_ecg[0, x, :].min(), original_ecg[0, x, :].max())

    # last row
    axes[12, 0].plot(range(0, original_ecg.shape[-1], 1), torch.zeros(size=(original_ecg.shape[-1],)), zorder=1) # (2500)
    sns.heatmap(importance_ecg[:, :].mean(dim=0, keepdim=True).repeat(15, 1), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 2500)
                alpha=0.5,
                zorder=2,
                ax=axes[12, 0])
    axes[12, 0].set_ylim(original_ecg.min(), original_ecg.max())

    # last column
    # get the grid of the last column
    gs = axes[0, -1].get_gridspec()
    # remove the underlying axes of the last column
    for ax in axes[:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:-1, -1])

    sns.heatmap(importance_ecg[:, :].mean(dim=-1, keepdim=True), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 12)
                alpha=0.75,
                ax=axbig)

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()

    plt.savefig(f'{save_dir}/local_ecg_{idx}.png', dpi=300)
    plt.close('all')



def plot_pairwise_localization(original_img,
                               original_ecg,
                               importance_pairwise,
                               idx: int,
                               num_frames: int = 50, save_dir: str = None):
    """Animate CMR frames & saliency while showing 12‑lead ECG in a **6×2 grid**.

    Lead order: Ⅰ, Ⅱ, Ⅲ, aVR, aVL, aVF, V1–V6 (row‑major).
    CMR + heat‑map occupies the bottom row spanning both columns.
    """

    lead_names = [
        "I", "Ⅱ", "Ⅲ",           # limb leads
        "aVR", "aVL", "aVF",      # augmented limb leads
        "V1", "V2", "V3", "V4", "V5", "V6"  # precordial
    ]

    # ---------------- shapes & prep ----------------
    B, T_img, H_img, W_img = original_img.shape
    _, _, C_sig, T_sig = original_ecg.shape
    _, NpC, H, W = importance_pairwise.shape
    assert C_sig == 12, "Expecting 12 ECG leads"

    # saliency → (B, num_frames, H', W')
    importance_pairwise = (
        importance_pairwise.flatten(2).permute(0, 2, 1)
    )
    # print(f'number of frames: {num_frames}')
    importance_pairwise = F.interpolate(importance_pairwise, size=num_frames, mode='linear')
    importance_pairwise = importance_pairwise.permute(0, 2, 1).view(B, num_frames, H, W)

    # normalise CMR 0‑1
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-6)
    importance_pairwise = 2 * (importance_pairwise - importance_pairwise.min()) / (importance_pairwise.max() - importance_pairwise.min()) - 1
    cmr_frames = original_img[idx].cpu()              # (T_img, H_img, W_img)
    saliency = importance_pairwise[idx].cpu()
    ecg_leads = original_ecg[idx, 0].cpu()           # (12, T_sig)

    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(7, 2, height_ratios=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 4],
                           hspace=0.05, wspace=0.1)

    # mapping from lead idx → subplot row/col
    ecg_axes = []
    for i in range(6):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            lead_idx = i * 2 + j
            t_axis = np.arange(T_sig)
            ax.plot(t_axis, ecg_leads[lead_idx], linewidth=0.6, color='black')
            ax.set_xlim(0, T_sig)
            ax.set_ylabel(lead_names[lead_idx], rotation=0, labelpad=15, fontsize=7, va='center')
            ax.tick_params(axis='both', which='both', length=0, labelsize=6)
            ax.set_yticks([])
            ax.set_xticks([])
            ecg_axes.append(ax)

    # highlight rectangles
    box_width = T_sig / num_frames
    rects = []
    for ax in ecg_axes:
        v_min, v_max = ax.get_ylim()
        rect = mpatch.Rectangle((0, v_min), box_width, v_max - v_min,
                                alpha=0.3, color='gray')
        ax.add_patch(rect)
        rects.append(rect)

    # CMR + saliency axis spans bottom row
    gs_cmr = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6, :], width_ratios=[20, 1], wspace=0.1)
    ax_img = fig.add_subplot(gs_cmr[0])
    base_disp = ax_img.imshow(cmr_frames[0], cmap='gray', aspect='auto', zorder=1)
    sal_disp = ax_img.imshow(np.zeros((H_img, W_img)), cmap='inferno', alpha=0.7,
                             zorder=2, vmin=saliency.min(), vmax=saliency.max())
    ax_img.axis('off')

    # Add colorbar in right sub‑column
    cax = fig.add_subplot(gs_cmr[1])
    cbar = fig.colorbar(sal_disp, cax=cax)
    cbar.set_label('Saliency', fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    # ---------------- animation ----------------
    def animate(frame_idx):
        x0 = (frame_idx / num_frames) * (T_sig - box_width)
        for rect in rects:
            rect.set_x(x0)
        # update CMR frame
        if frame_idx < cmr_frames.shape[0]:
            base_disp.set_data(cmr_frames[frame_idx])
        # update saliency
        imp = saliency[frame_idx]
        imp = F.interpolate(imp[None, None], size=(H_img, W_img),
                             mode='bilinear', align_corners=False).squeeze()
        sal_disp.set_data(imp)
        return rects + [base_disp, sal_disp]

    anim = animation.FuncAnimation(fig, animate,
                                   frames=num_frames,
                                   interval=50,
                                   blit=False,
                                   repeat=True)

    path = f'{save_dir}/local_ecgcmr_{idx}.gif'
    anim.save(path, writer='imagemagick', fps=max(num_frames // 10, 1))
    plt.close(fig)
    return path

