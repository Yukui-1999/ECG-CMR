import os
import torch
import numpy as np
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
import json
from CMR_generation.CMR_gen_models import mar, vae3d
from CMR_generation.CMR_gen_dataset import ECG_genCMR,ECGBaseMIMICDis,ECGyingguo,ECG_quzhougenCMR,ECG_zheergenCMR
import yaml
from models import encoder

def parse_tuple(value):
    return tuple(map(int, value.split(',')))

def get_args_parser():
    parser = argparse.ArgumentParser('parser', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # Dataset parameters
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/ecg_cmr_v1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/ecg_cmr_v1',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # mar config
    parser.add_argument('--gen_num', default=30000, type=int, help='debug mode')
    parser.add_argument('--diffloss_d', default=3, type=int, help='d')
    parser.add_argument('--diffloss_w', default=1024, type=int, help='w')
    parser.add_argument('--img_size', default=(50, 96, 96), type=parse_tuple, help='img_size')
    parser.add_argument('--vae_stride', default=(2, 8, 8), type=parse_tuple, help='vae_stride')
    parser.add_argument('--patch_size', default=(5, 2, 2), type=parse_tuple, help='patch_size')

    # phenotype vae config
    parser.add_argument('--latent_dim', default=32, type=int, help='z dim')
    parser.add_argument('--hidden_dims', default=(128, 128, 128), type=parse_tuple, help='hidden_dims')

    parser.add_argument('--mar_ckpt', default='/mnt/sda1/liziyu/ECG2CMR/output/mar_kl_ft2_fs8_z16_float16_checkpoint-0004_lr8e-4_ps522_v1/checkpoint-last.pth', type=str, help='mar model ckpt')
    parser.add_argument('--vae3d_config', default='/home/liziyu/CMRGEN/cmrmar/config/vae_kl_ft2_fs8_z16.yaml', type=str, help='vae3d model config yaml')
    parser.add_argument('--vae3d_ckpt', default='/mnt/sda1/liziyu/CMRMAR/output/vae_kl_ft2_fs8_z16_float16/checkpoint-0004.pth', type=str, help='vae3d model ckpt')
    parser.add_argument('--phenotype_vae_ckpt', default="", type=str, help='phenotype vae model ckpt')
    
    # generate config
    parser.add_argument('--num_ar_steps', default=16, type=int, help='generate ar step')
    parser.add_argument('--diff_temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--phenotype_vae_temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--cfg_scale', default=3.0, type=float, help='cfg scale')

    parser.add_argument('--use_rep_cond', action='store_true', help='use_rep_cond')
    # parser.set_defaults(use_rep_cond=True)
    parser.add_argument('--use_mae_loss', action='store_true', help='use_mae_loss')
    # parser.set_defaults(use_mae_loss=True)
    parser.add_argument('--coef_mae_loss', default=1.0, type=float, help='use_mae_loss')
    parser.add_argument('--diffloss_on_rep', action='store_true', help='扩散损失是否反向传播梯度到图像表征')

    parser.add_argument('--phenotype_path', default='None', type=str, help='是否提供采样好的指标')

    parser.add_argument('--sample_only', default=True, type=bool, help='采样样本，且不推理指标')
    parser.add_argument('--sample_data_path', default="/home/liziyu/CMRGEN/ECGCMR/test_data_v1.json", type=str, help='采样的集合')
    
    return parser

args = get_args_parser()
args = args.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
args.log_dir = args.output_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
if args.log_dir is not None:
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.log_dir + '/sample', exist_ok=True)
    os.makedirs(args.log_dir + '/config', exist_ok=True)

seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)

# config and init
###############################################################################
model_type = "mar_base" #@param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 100 #@param {type:"slider", min:1, max:1000, step:1}

model = mar.__dict__[model_type](
    buffer_size=64,
    diffloss_d=args.diffloss_d,
    diffloss_w=args.diffloss_w,
    num_sampling_steps=str(num_sampling_steps_diffloss),
    class_num=1,
    img_size=args.img_size,
    vae_stride=args.vae_stride,
    patch_size=args.patch_size,
    use_rep_cond=args.use_rep_cond,
    use_mae_loss=args.use_mae_loss,
    coef_mae_loss=args.coef_mae_loss,
    diffloss_on_rep=args.diffloss_on_rep,
).to(device)
# state_dict = torch.load("pretrained_models/mar/{}/checkpoint-last.pth".format(model_type))["model_ema"]
state_dict = torch.load(args.mar_ckpt)["model_ema"]
# print(model)
model.load_state_dict(state_dict, strict=False)
model.eval() # important!
del state_dict


model_config = OmegaConf.load(args.vae3d_config)

vae = vae3d.__dict__['vae3d'](
    lossconfig=model_config.model.params.lossconfig, 
    ddconfig=model_config.model.params.ddconfig, 
    ddconfig_2d=model_config.model.params.ddconfig_2d,
    embed_dim=model_config.model.params.embed_dim,
).to(device)
checkpoint = torch.load(args.vae3d_ckpt)
checkpoint = {
    key: value 
    for key, value in checkpoint.items() 
    if not key.startswith("loss.perceptual_loss")
}
vae.load_state_dict(checkpoint)
vae.eval()
del checkpoint


def build_ecg_model():
    
    with open(os.path.realpath('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align/configs/gen/st_mem.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = None
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

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
        print(msg)
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    ecg_model.to('cuda')
    ecg_model.eval()
    return ecg_model

ecg_model = build_ecg_model()
################################################################################

data = json.load(open(args.sample_data_path, 'r'))
ecg_dataset = ECGyingguo(data)
ecg_dataloader = DataLoader(ecg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True,drop_last=False)
print("ecg dataset length: ", len(ecg_dataset))
################################################################################

import numpy as np
import nibabel as nib
# Set user inputs:
num_ar_steps = args.num_ar_steps #@param {type:"slider", min:1, max:256, step:1}
cfg_scale = args.cfg_scale #@param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = args.diff_temperature #@param {type:"slider", min:0.9, max:1.1, step:0.01}

from tqdm import tqdm
# generate

iter = 0
pbar = tqdm(ecg_dataloader, desc="Processing")
for data in pbar:
    class_labels = data[0].to(device, non_blocking=True)
    class_labels = ecg_model(class_labels)
    file_name = data[1]
    
    not_exist_file_name_indicses = []
    for i in range(len(file_name)):
        if not os.path.exists(args.log_dir + '/sample/{}.nii'.format(file_name[i])):
            not_exist_file_name_indicses.append(i)
    file_name = [file_name[i] for i in not_exist_file_name_indicses]
    class_labels = class_labels[not_exist_file_name_indicses]
    # 实时更新进度条信息
    pbar.set_postfix({
        'valid_files': len(file_name),  # 显示当前批次的有效文件数量
        'dropped': args.batch_size - len(file_name),  # 显示当前批次丢弃的文件数量
    })
    
    if len(file_name) == 0:
        continue
    
    with torch.cuda.amp.autocast():
        sampled_tokens = model.sample_tokens(
            bsz=len(class_labels), num_iter=num_ar_steps,
            cfg=cfg_scale, cfg_schedule=cfg_schedule,
            labels=torch.Tensor(class_labels).float().cuda(),
            temperature=temperature, progress=False)
        sampled_images = vae.decode(sampled_tokens, None)
        sampled_images = sampled_images.clamp(-1.0, 1.0) # b 1 f h w
        # print(sampled_images.max(), sampled_images.min())

        sampled_images_np = sampled_images.cpu().float().numpy()
        for i in range(sampled_images_np.shape[0]):
            nii_image = nib.Nifti1Image(sampled_images_np[i, 0], np.eye(4))
            save_name = file_name[i]
            nib.save(nii_image, args.log_dir + '/sample/{}.nii'.format(save_name))

