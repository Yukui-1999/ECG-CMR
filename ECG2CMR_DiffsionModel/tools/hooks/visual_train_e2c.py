import os
import torch
import pynvml
import logging
from einops import rearrange
import torch.cuda.amp as amp
import SimpleITK as sitk
from utils.video_op import save_video_refimg_and_text
from utils.registry_class import VISUAL
import matplotlib.pyplot as plt
import numpy as np
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

@VISUAL.register_class()
class VisualTrainECGToCMR(object):
    def __init__(self, cfg_global, autoencoder, diffusion, partial_keys=[], guide_scale=9.0, use_offset_noise=None, **kwargs):
        super(VisualTrainECGToCMR, self).__init__(**kwargs)
        self.cfg = cfg_global
        self.diffusion = diffusion
        self.autoencoder = autoencoder
        self.guide_scale = guide_scale
        self.partial_keys_list = partial_keys
        self.use_offset_noise = use_offset_noise

    def prepare_model_kwargs(self, partial_keys, full_model_kwargs):
        """
        """
        model_kwargs = [{}, {}]
        for partial_key in partial_keys:
            model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
            model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]
        return model_kwargs
    
    @torch.no_grad()
    def run(self,
            model,
            video_data,
            captions,
            step=0,
            ref_frame=None,
            visual_kwards=[],
            **kwargs):
        
        cfg = self.cfg
        noise = torch.randn_like(video_data)
        # print(f'noise.shape:{noise.shape}')#(3, 4, 50, 32, 32)
        
        if self.use_offset_noise:
            noise_strength = getattr(cfg, 'noise_strength', 0)
            b, c, f, *_ = video_data.shape
            noise = noise + noise_strength * torch.randn(b, c, f, 1, 1, device=video_data.device)
        
        # import ipdb; ipdb.set_trace()
        # print memory
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')

        for keys in self.partial_keys_list:
            model_kwargs = self.prepare_model_kwargs(keys, visual_kwards)
            # print(f'model_kwargs:{model_kwargs}')
            pre_name = '_'.join(keys)
            with amp.autocast(enabled=cfg.use_fp16):
                video_data = self.diffusion.ddim_sample_loop(
                    noise=noise.clone(),
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=self.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
            
            video_data = 1. / cfg.scale_factor * video_data # [64, 4, 32, 48]
            video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
            chunk_size = min(cfg.decoder_bs, video_data.shape[0])
            video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size,dim=0)
            decode_data = []
            for vd_data in video_data_list:
                gen_frames = self.autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
            video_data = torch.cat(decode_data, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = ref_frame.shape[0])

            
            file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{pre_name}'
            local_path = os.path.join(cfg.log_dir, f'sample_{step:06d}/{file_name}')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            video_data = torch.mean(video_data,dim=1)
            ref_frame = torch.mean(ref_frame,dim=2)
            video_data = torch.clamp(video_data, -1.0, 1.0)

            # video_data = (video_data + 1.0) / 2.0
            video_data = (video_data - video_data.min()) / (video_data.max() - video_data.min())
            # ref_frame = (ref_frame + 1.0) / 2.0
            print(f'video_data:{video_data.shape}, ref_frame:{ref_frame.shape}')
            print(f'video_data:{video_data.min()}, {video_data.max()}')
            print(f'ref_frame:{ref_frame.min()}, {ref_frame.max()}')
            
            output_tensor = torch.cat((video_data, ref_frame), dim=2).to('cpu').numpy()
            for i in range(video_data.shape[0]):
                sitk.WriteImage(sitk.GetImageFromArray(output_tensor[i]), local_path + f'_{i}.nii.gz')
                plot_ecg(captions[i].squeeze().cpu().numpy(), file_path=local_path + f'_{i}.png')
            