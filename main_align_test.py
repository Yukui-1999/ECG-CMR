import os
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time
import datetime
import json
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from engine_align import train_one_epoch, evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGCMRTrain, ECGCMRValidation,ECGBasePhen,ECGCMRPhen
from util.plot_localization import plot_ecg_localization, plot_ecg_attention, plot_image_localization, plot_pairwise_localization

def get_args_parser():
    parser = argparse.ArgumentParser('CardioNets training ECG align with CMR', add_help=False)
    
    # model
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default='/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth', type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0.1, type=float, help='drop path rate')
    parser.add_argument('--cmr_fozen', default=True, type=bool, help='freeze cmr model')
    parser.add_argument('--ecg_pretrained_weights', default="/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align_update_cmrpca/ECG_STME_CMR_base_bs256_lr0.001_optadamw_seed42_latent256_clip0.05_sigma0.5_eps0.9_use_supconFalse/best-loss.pth", type=str, help='pretrained weights path')
    # /mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align/ECG_STME_CMR_base_bs256_lr0.001_optadamw_clip0.05_seed42/best-loss.pth
    # /mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align_update_cmrpca/ECG_STME_CMR_base_bs256_lr0.001_optadamw_seed42_latent256_clip0.05_sigma0.5_eps0.9_use_supconFalse/best-loss.pth
    # loss
    parser.add_argument('--clip_temperature', default=0.05, type=float, help='temperature for clip loss')
    
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align', type=str, help='number of classes')
    parser.add_argument('--exp_name', default='ECG_STME_CMR_base', type=str, help='number of classes')
    parser.add_argument('--exp_type', default='align', type=str, help='experiment type')
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--blr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-7, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--accum_iter', default=1, type=int, help='accumulation iterations')
    
    # training
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--latent_dim', default=256, type=int, help='latent dimension for ECG model')
    return parser



def build_ecg_model(args=None):
    
    with open(os.path.realpath('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/align/st_mem.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.exp_type == 'align':
        config['model']['num_classes'] = args.latent_dim
    else:
        config['model']['num_classes'] = None
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')
    if args.exp_type == 'scratch':
        return ecg_model
    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
        checkpoint_model = checkpoint['model']
        state_dict = ecg_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained ECG model: {msg}')
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    return ecg_model


def attention_forward_wrapper(attn_obj):
    """
    Modified version of def forward() of class Attention() in timm.models.vision_transformer
    """
    def my_forward(x):
        B, N, C = x.shape # C = embed_dim
        # print(f'B: {B}, N: {N}, C: {C}')
        # (3, B, Heads, N, head_dim)
        qkv = attn_obj.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=attn_obj.heads), qkv)

        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.dropout(attn)
        # (B, Heads, N, N)
        attn_obj.attn_map = attn # this was added 

        # (B, N, Heads*head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.to_out(x)
        return x
    return my_forward




def main(args):
    
    
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(args, default_flow_style=False, sort_keys=False))
    
    if args.exp_type == 'align':
        save_dir = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/align_attn/align_nejm_v1'
        args.cmr_pretrained_weights = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align_update_cmrpca/ECG_STME_CMR_base_bs256_lr0.001_optadamw_seed42_latent256_clip0.05_sigma0.5_eps0.9_use_supconFalse_addCMRmodelSave/best-loss-cmr.pth'
    elif args.exp_type == 'pretrain':
        save_dir = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/align_attn/pretrain'
    elif args.exp_type == 'Phen_sup':
        save_dir = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/align_attn/Phen_sup'
        args.ecg_pretrained_weights = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_yingguo_Reg/ECG_STME_CMR_base_Reg_scratch_with_phensup_bs128_lr5e-5_seed42_ECGmodescratch/best-r2.pth'
    elif args.exp_type == 'scratch':
        save_dir = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/align_attn/scratch'
    elif args.exp_type == 'align_not_freeze':
        save_dir = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/align_attn/align_not_freeze'
        args.ecg_pretrained_weights = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align/ECG_STME_CMR_base_bs256_lr0.001_optadamw_clip0.05_seed42_FreezeCMRFalse/best-loss.pth'
    else:
        raise ValueError(f'Unsupported experiment type: {args.exp_type}')
    os.makedirs(save_dir, exist_ok=True)
    # reproducibility
    seed = args.seed + misc.get_rank()
    print(f'seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    
    # dataset
    
    # valid_set = ECGCMRValidation()
    eid_json = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/val_ecgcmr_w_phelist_v1.json"
    eid = json.load(open(eid_json, 'r'))
    valid_set = ECGCMRPhen(txt_file=eid_json,isTrain=False)
    
    
    data_loader_valid = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"Valid dataset size: {len(valid_set)}")
    
    
    # log
    args.exp_name = args.exp_name + f'_bs{args.batch_size}_lr{args.blr}_opt{args.optimizer}_clip{args.clip_temperature}_seed{args.seed}_plot'
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
        
        
    # ECG model input shape (batchsize, 12, 2250)
    ECG_model = build_ecg_model(args)
    ECG_model.to(args.device)
    ECG_model.eval()
    if args.exp_type == 'align' or args.exp_type == 'align_not_freeze':
        checkpoint = torch.load(args.ecg_pretrained_weights, map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = ECG_model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained ECG model: {msg}')
    # depth is 12 block11 is the last
    ECG_model.block11.attn.fn.forward = attention_forward_wrapper(ECG_model.block11.attn.fn) # required to read out the attention map of the last layer
    
    # CMR model input shape (batchsize, 50, 96, 96)
    if args.exp_type == 'align':
        CMR_model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
            num_classes=args.latent_dim,
        )
    else:
        CMR_model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
        )
    checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')
    if args.exp_type != 'scratch':
        checkpoint_model = checkpoint['model']
        msg = CMR_model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained CMR model: {msg}')

    CMR_model.to(args.device)
    CMR_model.eval()
    

    

    umap_ecg_list = []
    umap_im_list = []
    # test
    for idx, (ecg, cmr,phen) in tqdm(enumerate(data_loader_valid), total=len(data_loader_valid), desc='Test'):
        if idx == 0:
            ecg = ecg.to(args.device, non_blocking=True)
            cmr = cmr.to(args.device, non_blocking=True)
            
            if ecg.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                latents_list = []
                attention_map_ecg_list = []
                for i in range(ecg.size(1)):
                    logits = ECG_model(ecg[:, i])
                    logits_list.append(logits)
                    latents_list.append(ECG_model.latent.reshape(args.batch_size, -1,ECG_model.latent.shape[-1]))
                    attention_map_ecg_list.append(ECG_model.block11.attn.fn.attn_map)
                logits_list = torch.stack(logits_list, dim=1)
                z_ecg = logits_list.mean(dim=1)
                umap_ecg_list.append(z_ecg.detach().cpu().numpy())
                z_local_ecg = torch.stack(latents_list, dim=1).mean(dim=1)
                #
                z_local_ecg = ECG_model.head(z_local_ecg)
                #
                attention_map_ecg = torch.stack(attention_map_ecg_list, dim=3).mean(dim=3)
                z_im = CMR_model(cmr)
                
                
                umap_im_list.append(z_im.detach().cpu().numpy())
                z_local_im = CMR_model.latent[:,1:,:]
                #
                z_local_im = CMR_model.head(z_local_im)
                #
                # print(f'Shape of z_ecg output: {z_ecg.shape}') #  torch.Size([32, 768]) 
                # print(f'Shape of z_im output: {z_im.shape}')#  torch.Size([32, 768])  
                # print(f'Shape of ECG model attention map: {attention_map_ecg.shape}') # torch.Size([32, 12, 384, 384])
                # print(f'Shape of z_local_ecg output: {z_local_ecg.shape}') # torch.Size([32, 360, 768])
                # print(f'Shape of z_local_im output: {z_local_im.shape}') # torch.Size([32, 36, 768])    
                z_im = F.normalize(z_im, dim=-1)
                z_local_im = F.normalize(z_local_im, dim=-1)
                # (B, H'*W', d)
                z_ecg = F.normalize(z_ecg, dim=-1)
                z_local_ecg = F.normalize(z_local_ecg, dim=-1)
                
                importance_ecg = torch.bmm(z_local_ecg, z_im.unsqueeze(-1)).squeeze(-1) / 0.05 # (32,360)
                # (B, C_sig, N'_(C_sig))
                importance_ecg = importance_ecg.view(args.batch_size, 12, -1) # (32,12,30)
                
                
                
                importance_im = torch.bmm(z_local_im, z_ecg.unsqueeze(-1)).squeeze(-1) / 0.05
                # (B, H', W')
                # print(f'Shape of importance map: {importance_im.shape}') # torch.Size([32, 36])
                importance_im = importance_im.view(args.batch_size, 6, 6)
                
                importance_pairwise = torch.bmm(z_local_ecg, z_local_im.transpose(1, 2)) / 0.05
                
                importance_pairwise = importance_pairwise.view(args.batch_size, 12, -1, 6, 6)
                # (B, N'_(C_sig), H', W')
                importance_pairwise = importance_pairwise.mean(1)
                # print(f'Shape of importance pairwise map: {importance_pairwise.shape}') #torch.Size([32, 30, 6, 6])
                # print(f'importance_ecg:{importance_ecg}')
                # print(f'importance_im:{importance_im}')
                # print(f'importance_pairwise:{importance_pairwise}') 
                
                # plot_ecg_attention(ecg.detach(), attention_map_ecg.detach(), index)
                for index in range(args.batch_size):
                    plot_ecg_localization(ecg.detach(), importance_ecg.detach(), index, save_dir)
                    plot_image_localization(cmr.detach(), importance_im.detach(), index, save_dir)
                    plot_pairwise_localization(cmr.detach(), ecg.detach(), importance_pairwise.detach(), index, save_dir=save_dir)

        else:
            break
            # with torch.no_grad():
            #     ecg = ecg.to(args.device, non_blocking=True)
            #     cmr = cmr.to(args.device, non_blocking=True)
                
            #     if ecg.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
            #         logits_list = []
                    
            #         for i in range(ecg.size(1)):
            #             logits = ECG_model(ecg[:, i])
            #             logits_list.append(logits)
                    
            #         logits_list = torch.stack(logits_list, dim=1)
            #         z_ecg = logits_list.mean(dim=1)
            #         umap_ecg_list.append(z_ecg.detach().cpu().numpy())
                
            #         z_im = CMR_model(cmr)
            #         umap_im_list.append(z_im.detach().cpu().numpy())
            #         del logits_list
    
    # umap_im = np.concatenate(umap_im_list, axis=0)
    # umap_ecg = np.concatenate(umap_ecg_list, axis=0)
    # print(f'Shape of umap_im: {umap_im.shape}') 
    # print(f'Shape of umap_ecg: {umap_ecg.shape}') 
    # combined = np.vstack([umap_ecg, umap_im])
    # labels = np.array([0] * len(umap_ecg) + [1] * len(umap_im))  # 0: ECG, 1: CMR

    # # ---------- 降维 ----------
    # print('Running UMAP...')
    # embedding_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)\
    #                     .fit_transform(combined)

    # print('Running t-SNE...')
    # embedding_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)\
    #                     .fit_transform(combined)

    # # ---------- 画图 ----------
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # colors = ['#1f77b4', '#ff7f0e']
    # modalities = ['ECG', 'CMR']

    # for ax, emb, title in zip(axes, [embedding_umap, embedding_tsne], ['UMAP', 't-SNE']):
    #     for label, color, name in zip([0, 1], colors, modalities):
    #         idx = labels == label
    #         ax.scatter(emb[idx, 0], emb[idx, 1], c=color, label=name, alpha=0.6, s=15)
    #     ax.set_title(f'{title} Projection')
    #     ax.set_xlabel(f'{title}-1')
    #     ax.set_ylabel(f'{title}-2')
    #     ax.legend()
    #     ax.grid(True)

    # plt.suptitle('Comparison of UMAP and t-SNE on ECG vs CMR Features')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f'comparison_umap_tsne_{args.exp_type}.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
