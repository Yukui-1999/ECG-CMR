import os
import yaml
import torch
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time
import datetime
import json
import pandas as pd
from tqdm import tqdm

from engine_Caldownstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import Example_ECGBaseMIMICDis
from util.val_result import process_val_result
def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
    
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--ecg_config_path', default='configs/align/example_st_mem.yaml', type=str, help='ecg config path')
    parser.add_argument('--ecg_pretrained_model', default="Example_downstreamTask/trained_by_MIMIC_CM_best-auc.pth", type=str, help='ecg pretrained model path')
    
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    # log
    parser.add_argument('--output_dir', default='Example_downstreamTask/', type=str, help='number of classes')
    
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cm', type=str, help='dis')
    parser.add_argument('--health_magnification', default=1, type=int, help='health magnification')
    parser.add_argument('--data_path', default="Example_downstreamTask/cm_mimic_test.pkl", type=str, help='data path')
    
    
    # training
    parser.add_argument('--device', default='cuda:0', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--best_patience', default=10, type=int, help='best patience')
    parser.add_argument('--train_percent', default=1, type=float, help='0.1,0.25,0.5,0.75,1')
    parser.add_argument('--use_gen_cmr', default=False, type=bool, help='use generated cmr')
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
    
    
    test_set = Example_ECGBaseMIMICDis(args.data_path)
        
    

    # dataloader

    data_loader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    

    
    # ECG model input shape (batchsize, 12, 2250)

    model = build_ecg_model(args.ecg_config_path)
    model.to(args.device)
    
    
    # log
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = None
        
    
    # loss
    ClaLoss = torch.nn.BCEWithLogitsLoss()


    
    
    print(f'begin testing')
    msg = model.load_state_dict(torch.load(args.ecg_pretrained_model)['model'])
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
    
    test_dir = output_dir
    
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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
