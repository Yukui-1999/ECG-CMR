import os
from sympy import N
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
import sys
from engine_align_update import train_one_epoch, evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import weighted_infonce_loss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGCMR_cmrPCAfeature


def setup_logger(log_file_path):
    class LoggerWriter:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = LoggerWriter(log_file_path)
    sys.stderr = sys.stdout  # 将错误也写入同一个日志文件

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
    parser = argparse.ArgumentParser('CardioNets training ECG align with CMR', add_help=False)
    
    # model
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default='/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth', type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0.1, type=float, help='drop path rate')
    parser.add_argument('--cmr_fozen', default=True, type=bool, help='freeze cmr model')
    
    # loss
    parser.add_argument('--tau', default=0.05, type=float, help='temperature for clip loss')
    parser.add_argument('--sigma', default=0.5, type=float, help='sigma for clip loss')
    parser.add_argument('--eps', default=0.9, type=float, help='eps for clip loss')
    parser.add_argument('--use_supcon', default=True, type=str2bool, help='use supcon loss')
    
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/ECG_CMR_align_update_cmrpca', type=str)
    parser.add_argument('--exp_name', default='ECG_STME_CMR_base', type=str, help='number of classes')
    
    # data
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
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
    parser.add_argument('--device', default='cuda:1', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--latent_dim', default=256, type=int, help='latent dimension for ECG model')
    return parser



def build_ecg_model(args=None):
    
    with open(os.path.realpath('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/align/st_mem.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = args.latent_dim
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
        print(f'Load pre-trained ECG model: {msg}')
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    return ecg_model

# ====== freeze backbone, keep head trainable ======
def freeze_cmr_backbone_but_keep_head_trainable(model):
    """
    冻结 VisionTransformer 主干参数，但保留 head 可训练。
    可选：将 backbone 设为 eval()（关闭 dropout 等），再把 head 切回 train()。
    """
    # 1) 先全部冻结
    for n, p in model.named_parameters():
        p.requires_grad = False

    # 2) 解冻 head
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Module):
        for n, p in model.head.named_parameters():
            p.requires_grad = True
        print("[CMR] Unfroze head params only.")
    else:
        raise AttributeError("CMR_model has no attribute 'head' or head is not a nn.Module.")


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
    train_set = ECGCMR_cmrPCAfeature(txt_file='/home/liziyu/CMRGEN/ECGCMR/train_data_v1.json',isTrain=True)
    valid_set = ECGCMR_cmrPCAfeature(txt_file='/home/liziyu/CMRGEN/ECGCMR/test_data_v1.json',isTrain=False)
    print(f'Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}')
    # debug
    # ecg,cmr,phen = train_set[0]
    # print(f'ECG shape: {ecg.shape}, CMR shape: {cmr.shape}, Phen shape: {phen.shape}')
    # ecg,cmr,phen = valid_set[0]
    # print(f'ECG shape: {ecg.shape}, CMR shape: {cmr.shape}, Phen shape: {phen.shape}')
    
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}")
    
    
    # log
    args.exp_name = args.exp_name + f'_bs{args.batch_size}_lr{args.blr}_opt{args.optimizer}_seed{args.seed}_latent{args.latent_dim}_clip{args.tau}_sigma{args.sigma}_eps{args.eps}_use_supcon{args.use_supcon}_addCMRmodelSave'
    output_dir = os.path.join(args.output_dir, args.exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
    log_file = os.path.join(output_dir, "train.log")
    setup_logger(log_file)
        
        
    # ECG model input shape (batchsize, 12, 2250)
    ECG_model = build_ecg_model(args=args)
    ECG_model.to(args.device)

    
    # CMR model input shape (batchsize, 50, 96, 96)
    CMR_model = models_vit.__dict__[args.cmr_model](
        drop_path_rate=args.drop_path,
        num_classes=args.latent_dim,
    )
    checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')

    checkpoint_model = checkpoint['model']
    msg = CMR_model.load_state_dict(checkpoint_model, strict=False)
    print(f'Load pre-trained CMR model: {msg}')
    if args.cmr_fozen:  
        print('Freeze CMR backbone (keep head trainable)')
        freeze_cmr_backbone_but_keep_head_trainable(CMR_model)
    
    num_total = sum(p.numel() for p in CMR_model.parameters())
    num_train = sum(p.numel() for p in CMR_model.parameters() if p.requires_grad)
    print(f"[CMR] Trainable params: {num_train} / {num_total}")
        
    CMR_model.to(args.device)
    
    
    
    # optimizer
    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.blr * eff_batch_size / 256.0
    print(f'Effective batch size: {eff_batch_size}, learning rate: {args.lr}')
    optim_params = list(ECG_model.parameters()) + list(CMR_model.head.parameters())
    optimizer = torch.optim.AdamW(optim_params,
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=args.weight_decay)
    
    
    # loss
    AlignLoss = weighted_infonce_loss(args=args)
    loss_scaler = NativeScaler()
    best_loss = float('inf')
    

    

    # Start training
    misc.load_model(vars(args), ECG_model, optimizer, loss_scaler)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    use_amp = args.use_amp
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(ECG_model,
                                      CMR_model,
                                      AlignLoss,
                                      data_loader_train,
                                      optimizer,
                                      args.device,
                                      epoch,
                                      loss_scaler,
                                      log_writer,
                                      vars(args),
                                      use_amp=use_amp,
                                      )

        valid_stats =  evaluate(ECG_model,
                                CMR_model,
                                AlignLoss,
                                data_loader_valid,
                                args.device,
                                epoch,
                                log_writer,
                                vars(args),
                                use_amp=use_amp,
                                )
        curr_loss = valid_stats['loss']                   
        if output_dir and curr_loss < best_loss:
            best_loss = curr_loss
            misc.save_model(vars(args),
                            os.path.join(output_dir, 'best-loss.pth'),
                            epoch,
                            ECG_model,
                            optimizer,
                            loss_scaler,
                            metrics={'loss': curr_loss})
            # save CMR model too
            misc.save_model(vars(args),
                                os.path.join(output_dir, 'best-loss-cmr.pth'),
                                epoch,
                                CMR_model,
                                )
    
    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + '\n')

    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
