'''
/* 
*Copyright (c) 2021, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
'''

import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
import pynvml
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import utils.transforms as data
from ..modules.config import cfg
from utils.util import to_device
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION, PRETRAIN, ECGCLIP, ECGCMRDATASET, ECGCLIPsa, ECGCMRDATASET_ECGlaCMR,ECGCMRDATASET_ECGCMR,ECGCMRDATASET_ECGlaCMRnew,ECGCMRDATASET_UKB_SMALL,ECGCMRDATASET_zheyi
import SimpleITK as sitk

@INFER_ENGINE.register_function()
def inference_ecg2cmr_entrance_zheyi(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        print('world size == 1')
        worker(0, cfg, cfg_update)
    else:
        print('world size != 1')
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg


def plot_ecg(ecg_data, fs=500, duration=10,file_path=None):
    """
    Plot a 12-lead ECG in a 6x2 grid.
    
    Parameters:
    ecg_data (numpy array): ECG data of shape (12, 5000)
    fs (int): Sampling frequency in Hz
    duration (int): Duration of the ECG signal in seconds
    """
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time = np.linspace(0, duration, ecg_data.shape[1])

    fig, axes = plt.subplots(6, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(time, ecg_data[i], label=leads[i])
        axes[i].legend(loc='upper right')
        axes[i].set_ylabel('mV')

    axes[-2].set_xlabel('Time (s)')
    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(file_path)
    # plt.show()

def plot_ecg_zheyi(ecg_data, fs=500, duration=10,file_path=None):
    """
    Plot a 12-lead ECG in a 6x2 grid.
    
    Parameters:
    ecg_data (numpy array): ECG data of shape (12, 5000)
    fs (int): Sampling frequency in Hz
    duration (int): Duration of the ECG signal in seconds
    """
    ecg_data = ecg_data * 0.005
    # 导联标签
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # 设置绘图参数
    fig, axs = plt.subplots(7, 2, figsize=(15, 10))
    # 绘制每个导联的前5秒数据
    for i in range(2):  # 6行，每行2个导联
        for j in range(6):
            lead_index = i * 6 + j
            time_axis = np.linspace(0, 5, 2500)
            axs[j, i].plot(time_axis, ecg_data[lead_index, :2500])  # 前5秒数据（2500个采样点）
            axs[j, i].set_title(leads[lead_index])
            axs[j, i].set_xlabel('Time (s)')
            axs[j, i].set_ylabel('Amplitude (mv)')
            axs[j, i].grid(True)

    # 合并最后一行的两个子图，用于绘制 II 导联 10秒数据
    axs_last = fig.add_subplot(7, 1, 7)
    time_axis_full = np.linspace(0, 10, 5000)  # 0到10秒，对应5000个采样点
    axs_last.plot(time_axis_full, ecg_data[1, :])  # 10秒数据（5000个采样点）
    axs_last.set_title('II')
    axs_last.set_xlabel('Time (s)')
    axs_last.set_ylabel('Amplitude (mv)')
    axs_last.grid(True)

    # 删除原来第7行的两个空子图
    for j in range(2):
        fig.delaxes(axs[6, j])

    # 调整布局以避免重叠
    plt.tight_layout()
    plt.savefig(file_path)
    # plt.show()

def worker(gpu, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    cfg = assign_signle_cfg(cfg, cfg_update, 'vldm_cfg')
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.test_list_path).split('.')[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    test_model = osp.basename(cfg.test_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (exp_name))
    os.makedirs(cfg.log_dir, exist_ok=True)
    exist_eid_list = [i.split('.')[0].split('_')[-2] + '_' + i.split('.')[0].split('_')[-1] for i in os.listdir(cfg.log_dir) if i.endswith('.nii.gz')]
    print(f'exist_eid_list:{exist_eid_list}')
    
    log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    cfg.log_file = log_file
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    logging.info(f"Going into inference_ecg2cmr_entrance inference on {gpu} gpu")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    # train_trans = data.Compose([
    #     data.CenterCropWide(size=cfg.resolution),
    #     data.ToTensor(),
    #     data.Normalize(mean=cfg.mean, std=cfg.std)])
    
    # vit_trans = data.Compose([
    #     data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
    #     data.Resize(cfg.vit_resolution),
    #     data.ToTensor(),
    #     data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    
    if cfg.ecgcmr_dataset.type == 'mutimodal_dataset_zheyi_ECG':
        dataset = ECGCMRDATASET_zheyi.build(cfg.ecgcmr_dataset)
    else:
        pass
   

    sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank) if (cfg.world_size > 1 and not cfg.debug) else None
    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    

    
    # [Model] embedder
    if cfg.ecgclip.type == 'ECGCLIPsa':
        ecgclip = ECGCLIPsa.build(cfg.ecgclip)
    elif cfg.ecgclip.type == 'ECGCLIP' or cfg.ecgclip.type == 'ECGEncoder':
        ecgclip = ECGCLIP.build(cfg.ecgclip)
    ecgclip.eval() # freeze
    for param in ecgclip.parameters():
        param.requires_grad = False
    ecgclip = ecgclip.to(gpu)

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    model = MODEL.build(cfg.UNet)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'state_dict' in state_dict:
        resume_step = state_dict['step']
        state_dict = state_dict['state_dict']
    else:
        resume_step = 0
    status = model.load_state_dict(state_dict, strict=True)
    logging.info('Load model from {} with status {}'.format(cfg.test_model, status))
    model = model.to(gpu)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()
    
   
    Eid_XmlPath = {
        'eid':[],
        'xml_path':[]
    }
    for idx, batch in enumerate(dataloader):
        print(f'idx:{idx}')
        batch = to_device(batch, gpu, non_blocking=True)
        eid = batch['eid']
        ori_ecg = batch['ori_ecg'].float()
        # if eid[0] in exist_eid_list:
        #     logging.info(f'{eid[0]} already exist, skip..')
        #     continue
        # if idx <= 142:
        #     print('exist')
        #     continue
        ecg = batch['ecg'].float()
        xml_path = batch['xml_path']
        
        cmr_encoder = torch.randn(ecg.shape[0], 4, 50, 32, 32)
        print(f'cmr_encoder:{cmr_encoder.shape}')
        # print(f'cmr.shape:{cmr.shape}')#cmr.shape:([batchsize, 50, 3, 256, 256]) 
        ecg = ecg.unsqueeze(1)
        
        with torch.no_grad():
            # _, y_text, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]
            if cfg.ecgclip.type == 'ECGCLIPsa':
                cond = batch['select_tar']
                y_words = ecgclip(ecg,cond)
                zero_y_negative = ecgclip(torch.randn_like(ecg),torch.randn_like(cond))
            elif cfg.ecgclip.type == 'ECGCLIP' or cfg.ecgclip.type == 'ECGEncoder':
                y_words = ecgclip(ecg)
                zero_y_negative = ecgclip(torch.randn_like(ecg))
            # zero_y_negative = None #试过不行报错
        # fps_tensor =  torch.tensor([cfg.target_fps], dtype=torch.long, device=gpu)

        with torch.no_grad():
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
            # sample images (DDIM)
            with amp.autocast(enabled=cfg.use_fp16):
                cur_seed = torch.initial_seed()
                logging.info(f"Current seed {cur_seed} ...")
                noise = torch.randn_like(cmr_encoder)
                # noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
                noise = noise.to(gpu)

                model_kwargs=[
                    {'y': y_words },
                    {'y': zero_y_negative}]
                # model_kwargs = {'y': y_words }
                video_data = diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
        
        video_data = 1. / cfg.scale_factor * video_data # [1, 4, 32, 46]
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = ecg.shape[0])
        
        
        file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:04d}'
        local_path = os.path.join(cfg.log_dir, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        video_data = torch.mean(video_data,dim=1)
        video_data = torch.clamp(video_data, -1.0, 1.0)
        video_data = (video_data - video_data.min()) / (video_data.max() - video_data.min())
        
        print(f'video_data:{video_data.shape}')
        output_tensor = video_data.to('cpu').numpy()
        for i in range(ecg.shape[0]):
            sitk.WriteImage(sitk.GetImageFromArray(output_tensor[i]), local_path + f'_{eid[i]}.nii.gz')
            plot_ecg_zheyi(ori_ecg[i].cpu().numpy(), file_path=local_path + f'_{eid[i]}_v2.png')
            Eid_XmlPath['eid'].append(local_path + f'_{eid[i]}')
            Eid_XmlPath['xml_path'].append(local_path + f'_{xml_path[i]}')
        
    
    logging.info('Congratulations! The inference is completed!')
    df = pd.DataFrame(Eid_XmlPath)
    # 将 DataFrame 写入 CSV 文件
    df.to_csv('/home/dingzhengyao/Work/T2Vdiff/VGen/zheyi_Eid_Xmlpath_ECGCMR_v2.csv', index=False)
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

