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
import pickle
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

import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler
import utils.transforms as data
from ..modules.config import cfg
from utils.util import to_device
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION, PRETRAIN, ECGCLIP, ECGCMRDATASET, ECGCLIPsa, ECGCMRDATASET_ECGlaCMR,ECGCMRDATASET_ECGCMR
import SimpleITK as sitk

@INFER_ENGINE.register_function()
def inference_ecg2cmr_entrance_generate(cfg_update,  **kwargs):
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


def save_data(rank, data, file_path):
    all_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_data, data)
    
    # 打印调试信息
    if rank == 0:
        print("all_data:", all_data)
    
    if rank == 0:
        # 检查并打印每个数据的格式
        if not all_data or not all_data[0]:
            print("Error: all_data is empty or improperly formatted")
            return
        # 保存List对象到文件
        # with open('list.pkl', 'wb') as file:
        #     pickle.dump(all_data, file)
        try:
            # 初始化合并后的字典
            print(f'all_data[0]:{all_data[0]}')
            print(f'all_data,lenth:{len(all_data)}')
            merged_data = {key: [] for key in all_data[0][0][1].keys()}
            print(f'merged_data:{merged_data.keys()}')
            merged_indices = []
            
            # 遍历收集到的每个进程的数据
            for sub_data in all_data:
                for index, sub_data_dict in sub_data:
                    merged_indices.append(index)
                    for key, value in sub_data_dict.items():
                        merged_data[key].append(value)

            merged_indices = np.concatenate([tensor.numpy() for tensor in merged_indices])
            for key, value in merged_data.items():
                if key == 'cmr':
                    reshaped_arrays = []
                    for arr in value:
                        if arr.ndim != 4:
                            reshaped_arrays.append(arr[np.newaxis, :])
                        else:
                            reshaped_arrays.append(arr)
                    merged_data[key] = np.concatenate(reshaped_arrays, axis=0)
                else: #ecg
                    reshaped_arrays = []
                    for arr in value:
                        if arr.ndim != 3:
                            reshaped_arrays.append(arr[np.newaxis, :])
                        else:
                            reshaped_arrays.append(arr)
                    merged_data[key] = np.concatenate(reshaped_arrays, axis=0)
            # 打印 merged_indices 内容进行调试
            print("merged_indices:", merged_indices)

            # 将每个 key 对应的列表合并为单个 ndarray，并按照原始索引排序
            sorted_indices = np.argsort(merged_indices)
            for key in merged_data.keys():
                merged_data[key] = np.array([merged_data[key][i] for i in sorted_indices])
            
            # 将合并后的数据保存为 .pt 文件
            torch.save(merged_data, file_path)
        except Exception as e:
            print(f"An error occurred during data merging: {e}")


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
    logging.info(f"Going into inference_ecg2cmr_entrance inference generate on {gpu} gpu")
    
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

    if cfg.select_cmr == 'cmr':
        dataset = ECGCMRDATASET_ECGCMR.build(cfg.ecgcmr_dataset)
    elif cfg.select_cmr == 'la_cmr':
        dataset = ECGCMRDATASET_ECGlaCMR.build(cfg.ecgcmr_dataset)

        # 定义每个子集的长度
    subset_length = 100

    # 计算子集的数量
    num_subsets = len(dataset) // subset_length

    # 创建子集列表
    subsets = [Subset(dataset, range(i * subset_length, (i + 1) * subset_length)) for i in range(num_subsets)]

    # 创建dataloader列表
    dataloaders = []
    print(f'num_subsets: {num_subsets}')
    # 遍历子集并创建对应的dataloader
    for subset in subsets:
        if cfg.world_size > 1 and not cfg.debug:
            sampler = DistributedSampler(subset, num_replicas=cfg.world_size, rank=cfg.rank)
        else:
            sampler = None

        if sampler is not None:
            dataloader = DataLoader(
                subset,
                sampler=sampler,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                prefetch_factor=cfg.prefetch_factor
            )
        else:
            # print('shuffle is false')
            dataloader = DataLoader(
                subset,
                shuffle=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                prefetch_factor=cfg.prefetch_factor
            )

        dataloaders.append(dataloader)

    # [Model] embedder
    # clip_encoder = EMBEDDER.build(cfg.embedder)
    # clip_encoder.model.to(gpu)
    # _, _, zero_y = clip_encoder(text="")
    # _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    # zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()
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
    
    # [Test List]
    # test_list = open(cfg.test_list_path).readlines()
    # test_list = [item.strip() for item in test_list]
    # num_videos = len(test_list)
    # logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    # test_list = [item for item in test_list for _ in range(cfg.round)]
    
    
    
    save_generate_data = {}
    print(f'cfg.cmr_generate_save_path:{cfg.cmr_generate_save_path}')
    logging.info('开始进入循环生成cmr')
    for load_idx,dataloader in enumerate(dataloaders):
        processed_data = []
        for idx, batch in enumerate(dataloader):
            generate_cmr = []
            ecg_val = []

            batch = to_device(batch, gpu, non_blocking=True)
            eid = batch['eid']
            ecg = batch['ecg']
            cond = batch['select_tar']
            
            ecg = ecg.unsqueeze(1)
            # print(f'ecg:{ecg.shape}, cmr:{cmr.shape}')
            cmr_encoder = torch.randn(ecg.shape[0], 4, 50, 32, 32)
            # print(f'cmr_encoder:{cmr_encoder.shape}')
            # if caption.startswith('#'):
            #     logging.info(f'Skip {caption}')
            #     continue
            # logging.info(f"[{idx}]/[{num_videos}] Begin to sample {caption} ...")
            # if caption == "": 
            #     logging.info(f'Caption is null of {caption}, skip..')
            #     continue
            # captions = [caption]
            with torch.no_grad():
                # _, y_text, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]
                if cfg.ecgclip.type == 'ECGCLIPsa':
                    y_words = ecgclip(ecg,cond)
                    zero_y_negative = ecgclip(torch.randn_like(ecg),torch.randn_like(cond))
                elif cfg.ecgclip.type == 'ECGCLIP'  or cfg.ecgclip.type == 'ECGEncoder':
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
            
            text_size = cfg.resolution[-1]
            # cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
            file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:04d}'
            local_path = os.path.join(cfg.log_dir, f'{file_name}')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            video_data = torch.mean(video_data,dim=1)
            # ref_frame = torch.mean(cmr,dim=2)
            video_data = torch.clamp(video_data, -1.0, 1.0)
            video_data = (video_data - video_data.min()) / (video_data.max() - video_data.min())
            
            video_data = video_data.to('cpu').numpy()
            for i in range(video_data.shape[0]):
                generate_cmr.append(video_data[i])
                ecg_val.append(ecg[i].squeeze().cpu().numpy())
            generate_cmr = np.array(generate_cmr).squeeze()
            ecg_val = np.array(ecg_val).squeeze()
            processed_data.append({'cmr': generate_cmr, 'ecg': ecg_val, 'eid': eid})
            # print(f'video_data:{video_data.shape}, ref_frame:{ref_frame.shape}')
            # output_tensor = torch.cat((video_data, ref_frame), dim=2).to('cpu').numpy()
            # for i in range(output_tensor.shape[0]):
            #     sitk.WriteImage(sitk.GetImageFromArray(output_tensor[i]), local_path + f'_{i}.nii.gz')
            #     plot_ecg(ecg[i].squeeze().cpu().numpy(), file_path=local_path + f'_{i}.png')
            # try:
            #     save_i2vgen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            #     logging.info('Save video to dir %s:' % (local_path))
            # except Exception as e:
            #     logging.info(f'Step: save text or video error with {e}')
        logging.info('Congratulations! saving generate!')
        new_path = f"{os.path.splitext(cfg.cmr_generate_save_path)[0]}_{load_idx}{os.path.splitext(cfg.cmr_generate_save_path)[1]}"

        if sampler is not None:
            save_data(cfg.rank, processed_data, new_path)
        else:
            torch.save(processed_data, new_path,pickle_protocol=4)
        # dataset.save_data(generate_cmr, cfg.cmr_generate_save_path)
        logging.info(f'Congratulations! The inference {load_idx} of {num_subsets} is completed!')

        # synchronize to finish some processes
        # if not cfg.debug:
        #     torch.cuda.synchronize()
        #     dist.barrier()

