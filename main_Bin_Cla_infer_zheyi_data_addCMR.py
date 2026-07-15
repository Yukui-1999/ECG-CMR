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

from engine_Caldownstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGzheyi_two_Base, ECGzheer_two_Base, ECGquzhou_two_Base, ECGzheyi_two_wGenCMR, ECGzheer_two_wGenCMR, ECGquzhou_two_wGenCMR,ECGzheyi_two_BaseCMR
from util.val_result import process_val_result
from ECG_genCMR_model import ECG_genCMR_model

def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
    
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/Cla/st_mem_align.yaml', type=str, help='ecg config path')
    parser.add_argument('--ecg_pretrained_model_path', default="/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_MIMIC_Cla/ECG_STME_Cls_bs32_lr5e-5_seed42_health_magnification1_ECGmodescratch_discm_trainPercent1/best-auc.pth", type=str, help='ecg pretrained model path')
    # "/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_MIMIC_Cla/ECG_STME_Cls_bs32_lr5e-5_seed42_health_magnification1_ECGmodescratch_discm_trainPercent1/best-auc.pth"
    # "/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_MIMIC_Cla/ECG_STME_Cls_bs32_lr5e-5_seed42_health_magnification1_ECGmodepretrain_discm_trainPercent1/best-auc.pth"
    # "/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_MIMIC_Cla/ECG_STME_Cls_bs32_lr5e-5_seed42_health_magnification1_ECGmodealign_discm_trainPercent1/best-auc.pth"
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_yingguo_Cla/ECG_STME_CMR_base_Cls_bs32_lr5e-5_seed42_health_magnification1_CMRmode_discm_fold0/best-auc.pth', type=str, help='pretrained weights path')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_modality', default='ECG', type=str, help='ECG or CMR')
    # log
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--test_dir_name', default='test', type=str, help='test dir name')
    
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--private_hospital', default='zheyi', type=str, choices=['zheyi', 'zheer', 'quzhou'], help='private hospital')
    
    
    
    # inference
   
    parser.add_argument('--device', default='cuda:0', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    
    return parser



def build_ecg_model(ecg_config_path):
    
    with open(os.path.realpath(ecg_config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = 1
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')


    return ecg_model



def main(args):
    
    
        
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(args, default_flow_style=False, sort_keys=False))
    if 'wGenCMR' in args.ecg_pretrained_model_path:
        args.output_dir = os.path.join(args.output_dir, args.ecg_pretrained_model_path.split('/')[-2]+'_wGenCMR')
        args.use_gen_cmr = True
    else:
        if args.input_modality == 'ECG':
            args.output_dir = os.path.join(args.output_dir, args.ecg_pretrained_model_path.split('/')[-2])
            args.use_gen_cmr = False
        elif args.input_modality == 'CMR':
            args.output_dir = os.path.join(args.output_dir, args.cmr_pretrained_weights.split('/')[-2])
            args.use_gen_cmr = False
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
    if args.private_hospital == 'zheer':
        if args.use_gen_cmr:
            test_set = ECGzheer_two_wGenCMR(isTrain=False)
        else:
            test_set = ECGzheer_two_Base(isTrain=False)
    elif args.private_hospital == 'quzhou': 
        if args.use_gen_cmr:
            test_set = ECGquzhou_two_wGenCMR(isTrain=False)
        else:
            test_set = ECGquzhou_two_Base(isTrain=False)
    elif args.private_hospital == 'zheyi':
        if args.use_gen_cmr:
            test_set = ECGzheyi_two_wGenCMR(isTrain=False)
        else:
            test_set = ECGzheyi_two_BaseCMR(isTrain=False,args=args)
    else:
        raise ValueError(f'Unsupported private hospital: {args.private_hospital}')
        
    print(f"test dataset size: {len(test_set)}")

    
    data_loader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    

    
    # ECG model input shape (batchsize, 12, 2250)
    if args.input_modality == 'ECG':
        if args.use_gen_cmr:
            model = ECG_genCMR_model(args)
            model.to(args.device)
        else:
            model = build_ecg_model(args.ecg_config_path)
            model.to(args.device)
    elif args.input_modality == 'CMR':
        model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
            num_classes=1,
        )
        checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')

        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained CMR model: {msg}')
        model.to(args.device)
    
    
    # log
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
        
    
    
    
    # loss
    ClaLoss = torch.nn.BCEWithLogitsLoss()


    
    print(f'begin testing')
    if args.input_modality == 'ECG':
        msg = model.load_state_dict(torch.load(args.ecg_pretrained_model_path)['model'])
    elif args.input_modality == 'CMR':
        msg = model.load_state_dict(torch.load(args.cmr_pretrained_weights)['model'], strict=False)
    print(f'load model: {msg}')
    model.eval()
    test_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                ClaLoss,
                                data_loader_test,
                                args.device,
                                log_writer,
                                use_amp=args.use_amp,
                                args=args,
                                )
    
    test_dir = os.path.join(output_dir, args.test_dir_name)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    args.metric_save_path = test_dir
    args.downtask_type = 'BCE'
    process_val_result(tgt_list, opt_list, args)
    test_log_stats = {f'test_{k}': v for k, v in test_stats.items()}
    
    if output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
            f.write(json.dumps(test_log_stats) + '\n\n')
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.private_hospital == 'zheyi':
        args.output_dir = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_zheyi_Bin_infer_addCMR/'
    elif args.private_hospital == 'zheer':
        args.output_dir = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_zheer_Bin_infer/'
    elif args.private_hospital == 'quzhou':
        args.output_dir = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_quzhou_Bin_infer/'
    else:
        raise ValueError(f'Unsupported private hospital: {args.private_hospital}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
