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
import pandas as pd
from tqdm import tqdm

from engine_Caldownstream import train_one_epoch, evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGBaseDis, CMRBaseDis, ECGDis_wGenCMR,renji_dataset
from ECG_genCMR_model import ECG_genCMR_model
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
  
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/Cla/st_mem_align.yaml', type=str, help='ecg config path')
    parser.add_argument('--ecg_pretrained_model', default="/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_MIMIC_Cla/ECG_STME_Cls_FirstRevision_nejm_v1_bs32_lr5e-5_seed42_health_magnification1_ECGmodealign_discm_trainPercent1/best-auc.pth", type=str, help='pretrained ecg model')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    # log
    parser.add_argument('--test_dir_name', default='renji', type=str, help='test dir name')
    
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cad', type=str, help='dis')
    parser.add_argument('--health_magnification', default=1, type=int, help='health magnification')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--data_path', default='/mnt/data2/ECG_CMR/zheyi_data/Final_data/renjishiyan.pkl', type=str, help='data path')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    
    return parser



def build_ecg_model(args):
    
    with open(os.path.realpath(args.ecg_config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = args.num_classes
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] == "pretrain":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
    elif config['mode'] == "align":
        checkpoint = torch.load(config['afterAlign_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['afterAlign_path']}")
    elif config['mode'] == "scratch":
        print('Training from scratch')
        return ecg_model
    else:
        raise ValueError(f'Unsupported mode: {config["mode"]}')
    
    # load pre-trained weights
    checkpoint_model = checkpoint['model']
    state_dict = ecg_model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Remove key {k} from pre-trained checkpoint")
            del checkpoint_model[k]
    msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
    print(f'Load pre-trained ECG model: {msg}')
    return ecg_model

def main(args):
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(args, default_flow_style=False, sort_keys=False))
    
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

    model = build_ecg_model(args)
    msg = model.load_state_dict(torch.load(args.ecg_pretrained_model, map_location='cpu')['model'], strict=False)
    print(f'Load pre-trained ECG model: {msg}')
    model.to(args.device)
    model.eval()
    
    # log
    output_dir = os.path.dirname(args.ecg_pretrained_model)
    output_dir = os.path.join(output_dir, args.test_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # data
    dataset = renji_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )
    
    
    # infer
    opt_list = np.array([])
    out_eid = []
    for index, (ecg, eid) in tqdm(enumerate(dataloader), total=len(dataloader), desc='infer'):
        sample = ecg.to(args.device)
        out_eid.extend(eid)
        with torch.cuda.amp.autocast():
            if sample.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(sample.size(1)):
                    logits = model(sample[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                output = logits_list.mean(dim=1)
            else:
                output = model(sample)
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output.detach()).cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output.detach()).cpu().numpy()])
    
    out = {
        'eid': out_eid,
        'opt': opt_list.tolist()
    }
    print(f'opt_list len: {len(opt_list)}')
    print(f'out_eid len: {len(out_eid)}')
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(output_dir, 'test_result.csv'), index=False)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)