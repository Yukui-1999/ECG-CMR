from visualizer import get_local
get_local.activate()
import argparse
from typing import Tuple
import numpy as np
import os
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import modeling.ECGEncoder_co as ECGEncoder
import modeling.ECGEncoder as ECGEncoder_noco
from util.mutimodal_dataset_sl import get_train_dataset_class,get_test_dataset_class
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.utils import resample
import pickle
import cv2
from utils import augmentations, transformations
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import gc

def grad_rollout(attentions, gradients, discard_ratio):
    print(f'attentions: {len(attentions)}')
    print(f'attentions[0]: {attentions[0].shape}')
    print(f'gradients: {len(gradients)}')
    print(f'gradients[0]: {gradients[0].shape}')
    result = torch.eye(attentions[0].size(-1))
    print(f'result: {result.shape}')
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    
    mask = result[0, 0 , 1 :]
    print(f'mask: {mask.shape}')
    # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    print(f'mask: {mask.shape}')
    mask = mask.reshape(12, 50).numpy()
    mask = mask / np.max(mask)
    print(f'mask: {mask.shape}')
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor,cond, category_index=5):
        self.model.zero_grad()
        _, output = self.model(input_tensor,cond)
        print(f'input_tensor: {input_tensor.shape}')
        print(f'output: {output.shape}')
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)


def visualize_attention_map(attention_rollout, original_image, gradient):
    # 将 attention rollout 和梯度信息结合
    attention_map = attention_rollout.squeeze().cpu().detach().numpy()
    gradient = gradient.squeeze().cpu().detach().numpy()
    print(f'attention_map:{attention_map.shape}')
    print(f'gradient:{gradient.shape}')
    # 通过梯度加权调整注意力图
    attention_map = attention_map * gradient

    # 归一化到 [0, 1]
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    print(f'attention_map:{attention_map.shape}')
    print(f'original_image:{original_image.shape}')
    # 将attention map 调整到与原图大小一致
    attention_map_resized = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))

    # 可视化
    plt.imshow(original_image)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.6)  # 将注意力图叠加在原图上
    plt.axis('off')
    plt.show()
    plt.savefig('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/Visualizer/test_attn.png')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_args_parser():
    parser = argparse.ArgumentParser('ECG finetune test', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    
    # downstream task
    parser.add_argument('--downstream', default='classification', type=str, help='downstream task')
    parser.add_argument('--regression_dim', default=82, type=int, help='regression_dim')
    parser.add_argument('--classification_dis', default='I42', type=str, help='classification_dis')
    parser.add_argument('--resizeshape',default=256,type=int,help='resize shape')
    # Model parameters
    parser.add_argument('--latent_dim', default=2048, type=int, metavar='N',
                        help='latent_dim')
   
    # ECG Model parameters
    parser.add_argument('--threshold', default=0.5, type=float,help='threshold')
    parser.add_argument('--ecg_model', default='vit_large_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',
                        default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG/ep400_40_lr1e-4_bs10_wd0.05_regression_EF_freezeFalse/checkpoint-27-correlation-0.42.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--ecg_input_channels', type=int, default=1, metavar='N',
                        help='ecginput_channels')
    parser.add_argument('--ecg_input_electrodes', type=int, default=12, metavar='N',
                        help='ecg input electrodes')
    parser.add_argument('--ecg_time_steps', type=int, default=5000, metavar='N',
                        help='ecg input length')
    parser.add_argument('--ecg_input_size', default=(12, 5000), type=Tuple,
                        help='ecg input size')
    parser.add_argument('--ecg_patch_height', type=int, default=1, metavar='N',
                        help='ecg patch height')
    parser.add_argument('--ecg_patch_width', type=int, default=100, metavar='N',
                        help='ecg patch width')
    parser.add_argument('--ecg_patch_size', default=(1, 100), type=Tuple,
                        help='ecg patch size')
    parser.add_argument('--ecg_globle_pool', default=False, type=str2bool, help='ecg_globle_pool')
    parser.add_argument('--ecg_drop_out', default=0.0, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--ECGencoder_withco', default=True, type=str2bool,help='with_co or not')
    # Augmentation parameters
    parser.add_argument('--input_size', type=tuple, default=(12, 5000))

    parser.add_argument('--timeFlip', type=float, default=0.33)

    parser.add_argument('--signFlip', type=float, default=0.33)
    parser.add_argument('--condition_dim', default=24, type=int)
    # Dataset parameters
    parser.add_argument('--dataset',default='mutimodal_dataset_ECG',type=str)
    parser.add_argument('--train_data_path',
                        default="/mnt/data2/ECG_CMR/trainval_onlyecg_data_dict_v11_dn.pt",
                        type=str,
                        help='dataset path')
    parser.add_argument('--training_percentage',default=1.0,type=float)
    parser.add_argument('--test_data_path',
                        default="/mnt/data2/ECG_CMR/test_onlyecg_data_dict_v11.pt",
                        type=str,
                        help='test dataset path')
    
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--scaler', default="/mnt/data2/ECG_CMR/Cha_scaler_v11_dn.pkl", type=str)
    parser.add_argument('--save_dir', default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts", type=str)
    parser.add_argument('--input_data', default='LVM_max', type=str)

    return parser

def compute_attention_rollout(attn_list, discard_ratio=0.0):
    """
    计算 attention rollout.
    
    参数:
    - attn_list: List[Tensor]，形状为 (1, 16, 601, 601)，共有24个，分别来自每一层的注意力矩阵。
    - discard_ratio: 用于剪枝低值注意力权重的比例，默认为0，即不进行剪枝。

    返回:
    - attention_rollout: 最终的全局注意力矩阵，形状为 (1, 601, 601)。
    """
    # 初始化rollout为第一层的注意力矩阵，先对head求平均
    # 形状从 (1, 16, 601, 601) -> (1, 601, 601)
    rollout = attn_list[0].mean(dim=1)

    for i in range(1, len(attn_list)):
        # 对每层的head求平均，(1, 16, 601, 601) -> (1, 601, 601)
        current_attn = attn_list[i].mean(dim=1)

        # 如果需要，可以去掉最小的部分attention值
        if discard_ratio > 0:
            flat = current_attn.flatten(1)
            _, indices = flat.topk(int(flat.size(-1) * (1 - discard_ratio)), dim=-1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, indices, 1)
            current_attn = current_attn * mask.view_as(current_attn)

        # 将当前层的注意力矩阵与之前的rollout相乘，累积注意力信息
        rollout = torch.matmul(current_attn, rollout)

    return rollout

def main(args):
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction']
    # select_index = [3]
    model = ECGEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.regression_dim,
                drop_rate=args.ecg_drop_out,
                condition_dim=args.condition_dim,
                args=args,
            )
    # ecg_pretrained_model = "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse_nopretrained/checkpoint-40-correlation-0.35.pth"
    

    # if args.scaler is not None:
    #     train_scaler = pickle.load(open(args.scaler, 'rb'))
    #     dataset_test = get_test_dataset_class(args.dataset,args,train_scaler)
    # else:
    #     dataset_test = get_test_dataset_class(args.dataset,args,None)

    # MinMax_indicators = dataset_test.get_indicator()
    # pickle.dump(MinMax_indicators, open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/MinMax_indicators18.pkl', 'wb'))
    data_key = {'LA ejection fraction_min': '4481404', 'LA ejection fraction_max': '5053584', 'LA maximum volume_max': '1389542', 'LA maximum volume_min': '1744595', 'LA minimum volume_max': '1824707', 'LA minimum volume_min': '4802474', 'LA stroke volume_max': '2631836', 'LA stroke volume_min': '5752374', 'LV cardiac output_max': '1450021', 'LV cardiac output_min': '5900185', 'LV ejection fraction_max': '1368894', 'LV ejection fraction_min': '4574485', 'LV end diastolic volume_max': '3425779', 'LV end diastolic volume_min': '4353687', 'LV end systolic volume_min': '2835746', 'LV end systolic volume_max': '5074341', 'LV myocardial mass_min': '1164285', 'LV myocardial mass_max': '1737775', 'LV stroke volume_max': '2909049', 'LV stroke volume_min': '4867182', 'RA ejection fraction_max': '4787648', 'RA ejection fraction_min': '5117324', 'RA maximum volume_max': '2553305', 'RA maximum volume_min': '3023453', 'RA minimum volume_max': '1997717', 'RA minimum volume_min': '5670806', 'RA stroke volume_min': '2015735', 'RA stroke volume_max': '2029274', 'RV ejection fraction_min': '2522256', 'RV ejection fraction_max': '4564849', 'RV end diastolic volume_min': '1529243', 'RV end diastolic volume_max': '5405029', 'RV end systolic volume_min': '1079516', 'RV end systolic volume_max': '3212043', 'RV stroke volume_max': '2107920', 'RV stroke volume_min': '3141557'}
    # LVM_obj = dataset_test.get_LVM_maxmin()
    # RVEDV_obj = dataset_test.get_RVEDV_maxmin()

    # LVM_max_eid = LVM_obj['LVM_max_eid']
    # LVM_min_eid = LVM_obj['LVM_min_eid']
    # RVEDV_min_eid = RVEDV_obj['RVEDV_min_eid']
    # RVEDV_max_eid = RVEDV_obj['RVEDV_max_eid']
    # print(f'LVM_max_eid: {LVM_max_eid}')
    # print(f'LVM_min_eid: {LVM_min_eid}')
    # print(f'RVEDV_min_eid: {RVEDV_min_eid}')
    # print(f'RVEDV_max_eid: {RVEDV_max_eid}')
    # LVM_max_data = {
    #     'eid': LVM_obj['LVM_max_eid'][1],
    #     'ecg': LVM_obj['LVM_max_ecg'][1],
    #     'select_tar': LVM_obj['LVM_max_select_tar'][1],
    #     'data': LVM_obj['LVM_max'][1],
    # }
    # pickle.dump(LVM_max_data, open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/LVM_max_data.pkl', 'wb'))
    # LVM_min_data = {
    #     'eid': LVM_obj['LVM_min_eid'][6],
    #     'ecg': LVM_obj['LVM_min_ecg'][6],
    #     'select_tar': LVM_obj['LVM_min_select_tar'][6],
    #     'data': LVM_obj['LVM_min'][6],
    # }
    # pickle.dump(LVM_min_data, open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/LVM_min_data.pkl', 'wb'))
    # RVEDV_max_data = {
    #     'eid': RVEDV_obj['RVEDV_max_eid'][6],
    #     'ecg': RVEDV_obj['RVEDV_max_ecg'][6],
    #     'select_tar': RVEDV_obj['RVEDV_max_select_tar'][6],
    #     'data': RVEDV_obj['RVEDV_max'][6],
    # }
    # pickle.dump(RVEDV_max_data, open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/RVEDV_max_data.pkl', 'wb'))
    # RVEDV_min_data = {
    #     'eid': RVEDV_obj['RVEDV_min_eid'][8],
    #     'ecg': RVEDV_obj['RVEDV_min_ecg'][8],
    #     'select_tar': RVEDV_obj['RVEDV_min_select_tar'][8],
    #     'data': RVEDV_obj['RVEDV_min'][8],
    # }
    # pickle.dump(RVEDV_min_data, open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/RVEDV_min_data.pkl', 'wb'))
    # exit()

    # input_data = pickle.load(open(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/',args.input_data + '_data.pkl'), 'rb'))
    
    # colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #           'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    #           'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone','pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #           'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #           'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv',
    #           'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
    #           'tab20c','flag', 'prism', 'ocean', 'gist_earth', 'terrain','gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #            'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']   
    
    with open('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/MinMax_indicators18.pkl', 'rb') as f:
        input_data = pickle.load(f)

    csv_file = {}
    csv_file['min_eid'] = input_data['min_eid']
    csv_file['max_eid'] = input_data['max_eid']
    df = pd.DataFrame(csv_file)
    df.to_csv('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/output.csv', index=False)

    for i in range(18):
        print(cor_index[i])
        print('min')
        print(input_data['min_eid'][i])
        print('max')
        print(input_data['max_eid'][i])
    # exit()

    ecg_pretrained_model = "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse/checkpoint-29-correlation-0.43.pth"
    ecg_checkpoint = torch.load(ecg_pretrained_model, map_location='cpu')
    ecg_checkpoint_model = ecg_checkpoint['model']
    msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
    model.eval()
    print(msg)
    
    for z in range(18):
        
        for k in range(2):
            # if not z in select_index:
            #     print(f'jump')
            #     continue
            # j = 7
            if k == 0:
                eid_list = [str(i) for i in input_data['min_eid'][z]]
                eid_list = [x.replace('tensor(', '').replace(')', '') for x in eid_list]
                j = eid_list.index(data_key[cor_index[z]+'_min'])
                save_name = cor_index[z] + '_' + str(input_data['min_eid'][z][j]) + '_' + 'min.png'
                ecg = input_data['min_ecg'][z][j]
                original_ecg = ecg 
                condition_tensor = input_data['min_select_tar'][z][j]
            elif k == 1:
                eid_list = [str(i) for i in input_data['max_eid'][z]]
                eid_list = [x.replace('tensor(', '').replace(')', '') for x in eid_list]
                j = eid_list.index(data_key[cor_index[z]+'_max'])
                save_name = cor_index[z] + '_' + str(input_data['max_eid'][z][j]) + '_' + 'max.png'
                ecg = input_data['max_ecg'][z][j]
                original_ecg = ecg 
                condition_tensor = input_data['max_select_tar'][z][j]

            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
            ecg = ecg.unsqueeze(0).unsqueeze(0)
            print(ecg.shape)
            condition_tensor = condition_tensor.unsqueeze(0)
            input_tensor = ecg

            grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)
            mask = grad_rollout(input_tensor,condition_tensor,z)
            print(f'mask shape: {mask.shape}')

            mask = cv2.resize(mask, (5000, 12))
            print(f'mask: {mask.shape}')
        
            ecg_signal = np.array(original_ecg)
            heatmap = np.array(mask)

            # 创建一个图形
            fig, ax = plt.subplots(12, 1, figsize=(15, 15), sharex=True)

            for i in range(12):
                # 先画热力图，使用imshow
                ax[i].imshow(np.tile(heatmap[i], (100, 1)), aspect='auto', cmap='YlGnBu', extent=[0, 5000, np.min(ecg_signal[i]), np.max(ecg_signal[i])])
                # 叠加ECG信号曲线
                ax[i].plot(np.arange(5000), ecg_signal[i], color='black')

                # 设置y轴范围为ECG信号的最大最小值，便于可视化
                ax[i].set_ylim([np.min(ecg_signal[i]), np.max(ecg_signal[i])])

            # 设置x轴标签
            # ax[-1].set_xlabel('Time (ms)')
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/Indicators_correction',save_name))
            plt.close(fig) 
            del mask, ecg_signal, heatmap, ecg, condition_tensor
            gc.collect()

    ecg_pretrained_model = "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse_nopretrained/checkpoint-40-correlation-0.35.pth"
    ecg_checkpoint = torch.load(ecg_pretrained_model, map_location='cpu')
    ecg_checkpoint_model = ecg_checkpoint['model']
    msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
    model.eval()
    print(msg)
    
    for z in range(18):
        
        for k in range(2):
            # if not z in select_index:
            #     print(f'jump')
            #     continue
            # j = 7
            if k == 0:
                eid_list = [str(i) for i in input_data['min_eid'][z]]
                eid_list = [x.replace('tensor(', '').replace(')', '') for x in eid_list]
                j = eid_list.index(data_key[cor_index[z]+'_min'])
                save_name = cor_index[z] + '_' + str(input_data['min_eid'][z][j]) + '_' + 'min_nopretrained.png'
                ecg = input_data['min_ecg'][z][j]
                original_ecg = ecg 
                condition_tensor = input_data['min_select_tar'][z][j]
            elif k == 1:
                eid_list = [str(i) for i in input_data['max_eid'][z]]
                eid_list = [x.replace('tensor(', '').replace(')', '') for x in eid_list]
                j = eid_list.index(data_key[cor_index[z]+'_max'])
                save_name = cor_index[z] + '_' + str(input_data['max_eid'][z][j]) + '_' + 'max_nopretrained.png'
                ecg = input_data['max_ecg'][z][j]
                original_ecg = ecg 
                condition_tensor = input_data['max_select_tar'][z][j]
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
            ecg = ecg.unsqueeze(0).unsqueeze(0)
            print(ecg.shape)
            condition_tensor = condition_tensor.unsqueeze(0)
            input_tensor = ecg

            grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)
            mask = grad_rollout(input_tensor,condition_tensor,z)
            print(f'mask shape: {mask.shape}')

            mask = cv2.resize(mask, (5000, 12))
            print(f'mask: {mask.shape}')
        
            ecg_signal = np.array(original_ecg)
            heatmap = np.array(mask)

            # 创建一个图形
            fig, ax = plt.subplots(12, 1, figsize=(15, 15), sharex=True)

            
            for i in range(12):
                # 先画热力图，使用imshow
                ax[i].imshow(np.tile(heatmap[i], (100, 1)), aspect='auto', cmap='YlGnBu',extent=[0, 5000, np.min(ecg_signal[i]), np.max(ecg_signal[i])])
                # 叠加ECG信号曲线
                ax[i].plot(np.arange(5000), ecg_signal[i], color='black')

                # 设置y轴范围为ECG信号的最大最小值，便于可视化
                ax[i].set_ylim([np.min(ecg_signal[i]), np.max(ecg_signal[i])])

            # 设置x轴标签
            # ax[-1].set_xlabel('Time (ms)')
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/attentionMap/Indicators_correction',save_name))
            plt.close(fig) 
            del mask, ecg_signal, heatmap, ecg, condition_tensor
            gc.collect()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)