# Modified by Lang Huang (laynehuang@outlook.com)
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Tuple
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.mutimodal_dataset_sl import get_train_dataset_class,get_test_dataset_class
import timm.optim.optim_factory as optim_factory
from modeling.Config import Config_swin_base,Config_swin_base_win14,Config_swin_large_win14
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchsampler import ImbalancedDatasetSampler
from utils.callbacks import EarlyStop
from modeling.swin_transformer import build_swin as build_swin_woCond
from util.extract_backbone import load_pretrained
from engine_CMRsup import train_one_epoch,evaluate
from modeling import get_pretrain_model
import modeling.ECGEncoder_co as ECGEncoder
from modeling.swin_transformer_other import build_swin
from modeling.swin_transformer_both import SwinTransformerBoth

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
    parser = argparse.ArgumentParser('CMR_sup', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--resizeshape',default=224,type=int,help='resize shape')
    # Model parameters
    parser.add_argument('--cmr_mode',default='cmr',type=str,help='CMR_mode')
    parser.add_argument('--model', default='green_mim_swin_base_patch4_dec512b1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained_swin', default=None, type=str, metavar='PATH',)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--train_data_path',
                        default="/mnt/data2/ECG_CMR/trainval_data_dict_v11.pt",
                        type=str,
                        help='dataset path')
    parser.add_argument('--test_data_path',
                        default="/mnt/data2/ECG_CMR/test_data_dict_v11.pt",
                        type=str,
                        help='test dataset path')
    parser.add_argument('--dataset',default='mutimodal_dataset_laCMR',type=str)
    
    parser.add_argument('--cmr_isreal',default=True,type=str2bool)
    parser.add_argument('--with_cond',default=False,type=str2bool)
    parser.add_argument('--timeFlip', type=float, default=0.33)
    parser.add_argument('--signFlip', type=float, default=0.33)
    parser.add_argument('--ecg_input_size', type=tuple, default=(12, 5000))
    parser.add_argument('--ecg_patch_size', default=(1, 100), type=Tuple,help='ecg patch size')
    parser.add_argument('--ecg_input_channels', type=int, default=1, metavar='N',
                        help='ecginput_channels')
    parser.add_argument('--ecg_drop_out', default=0.05, type=float)
    #downstream task
    parser.add_argument('--downstream', default='regression', type=str, help='downstream task')
    parser.add_argument('--regression_dim',default=82,type=int,help='regression_dim')
    parser.add_argument('--classification_dis',default='I21',type=str,help='classification_dis')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--patience', default=10, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0.005, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/",
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:2',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--select_modal',default='la_cmr_data')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(f'device:{device}')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    if args.model == 'green_mim_swin_base_patch4_dec512b1':
        cmr_model_config = Config_swin_base()
    if args.model == 'green_mim_swin_base_patch4_win14_dec512b1':
        cmr_model_config = Config_swin_base_win14()
    if args.model == 'green_mim_swin_large_patch4_win14_dec512b1':
        cmr_model_config = Config_swin_large_win14()

    print('load dataset')
    if args.downstream == 'yaxing' or args.dataset == 'mutimodal_dataset_zheyi_ECG' or args.dataset == 'mutimodal_dataset_MIMIC_ECG' or args.dataset == 'mutimodal_dataset_NEWECG'or args.dataset == 'mutimodal_dataset_MIMIC_NEWECG':
        dataset_train = get_train_dataset_class(args.dataset,args)
        dataset_test = get_test_dataset_class(args.dataset,args)
    else:
        dataset_train = get_train_dataset_class(args.dataset,args)
        train_scaler = dataset_train.get_scaler()
        dataset_test = get_test_dataset_class(args.dataset,args,train_scaler)

    print(f'train set len:{len(dataset_train)}')
    print(f'test set len:{len(dataset_test)}')

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    
    if args.downstream == 'regression' or  args.downstream == 'yaxing' or args.dataset == 'mutimodal_dataset_zheyi_ECG' or args.dataset == 'mutimodal_dataset_MIMIC_ECG' or args.dataset == 'mutimodal_dataset_NEWECG' or args.dataset == 'mutimodal_dataset_MIMIC_NEWECG':
        if args.downstream == 'yaxing':
            args.latent_dim = args.yangxing_classes
        if args.downstream == 'classification' and ( args.dataset == 'mutimodal_dataset_zheyi_ECG'  or args.dataset == 'mutimodal_dataset_MIMIC_ECG' or args.dataset == 'mutimodal_dataset_NEWECG' or args.dataset == 'mutimodal_dataset_MIMIC_NEWECG'):
            cmr_model_config.MODEL.NUM_CLASSES = 1
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        train_sampler = ImbalancedDatasetSampler(dataset_train)
        cmr_model_config.MODEL.NUM_CLASSES = 1
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )
    if args.cmr_mode == 'cmr' or args.cmr_mode == 'la_cmr':
        if args.with_cond:
            model = build_swin(cmr_model_config)
        else:
            model = build_swin_woCond(cmr_model_config)
        model.to(device)
        if args.pretrained_swin:
            load_pretrained(args.pretrained_swin,model)
    else:
        model = SwinTransformerBoth(config=cmr_model_config,args=args)
        model.to(device)

    # print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    
    
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =args.epochs) #  * iters 

    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    eval_criterion = "correlation"
    best_stats = {'correlation': -np.inf}
    if args.downstream == 'classification':
        eval_criterion = "auc"
        best_stats = {'auc': -np.inf}
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # profiler.step()

        test_stats = evaluate(
            model, data_loader_test,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if eval_criterion == "loss":
                if early_stop.evaluate_decreasing_metric(val_metric=test_stats[eval_criterion]):
                    break
                if args.output_dir and test_stats[eval_criterion] <= best_stats[eval_criterion]:
                    misc.save_best_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion)
                    best_stats[eval_criterion] = test_stats[eval_criterion]
        else:
            if early_stop.evaluate_increasing_metric(val_metric=test_stats[eval_criterion]):
                break
            if args.output_dir and test_stats[eval_criterion] >= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion)
                best_stats[eval_criterion] = test_stats[eval_criterion]

        if args.downstream == 'classification':
            best_stats['auc'] = max(best_stats['auc'], test_stats['auc'])
        else:
            best_stats['correlation'] = max(best_stats['correlation'], test_stats['correlation'])
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
