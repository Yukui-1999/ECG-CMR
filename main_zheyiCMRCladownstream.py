import os
import pickle
import test
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

from engine_Caldownstream import train_one_epoch, evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGBaseDis, CMRBaseDis, ECGDis_wGenCMR,ECGzheyi_two_BaseCMR
from ECG_genCMR_model import ECG_genCMR_model
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default=None, type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--input_modality', default='ECG', type=str, help='ECG or CMR')
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/Cla/st_mem_align.yaml', type=str, help='ecg config path')
    parser.add_argument('--use_gen_cmr', default=False, type=bool, help='use generated cmr')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_Rework_v1/', type=str, help='number of classes')
    parser.add_argument('--test_dir_name', default='test', type=str, help='test dir name')
    parser.add_argument('--val_dir_name', default='val', type=str, help='test dir name')
    parser.add_argument('--train_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/nejm_revision_zheyi_cmr/zheyi_data_final3_minus_renji.pkl', type=str, help='train path')
    parser.add_argument('--test_path',default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/nejm_revision_zheyi_cmr/renjishiyan_addCMR.pkl', type=str, help='test path')
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cad', type=str, help='dis')
    parser.add_argument('--health_magnification', default=1, type=int, help='health magnification')
    
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--blr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--accum_iter', default=1, type=int, help='accumulation iterations')
    
    # training
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--best_patience', default=10, type=int, help='best patience')
    parser.add_argument('--fold', default=None, type=int, help='fold 0,1,2,3,4')
    parser.add_argument('--only_test', action='store_true', help='only test')

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
def split_five_fold(data_dict, fold, seed=42):
    """
    五折交叉验证（按 label 分层）：
      - 将 data_dict['label'] 非0视为1，仅用于划分；
      - 1类数量较少时，均匀分配到五折中；
      - 其余键的样本顺序对应关系保持不变。
    参数:
        data_dict: dict, 每个 key -> list，长度一致，且包含 'label'
        fold: int, [0-4] 指定哪一折作为验证集
        seed: int, 随机种子
    返回:
        train_dict, test_dict
    """
    assert 'label' in data_dict, "data_dict 必须包含 'label' 键"

    n = len(next(iter(data_dict.values())))
    assert all(len(v) == n for v in data_dict.values()), "dict 各个 key 的长度不一致"
    assert 0 <= fold < 5, "fold 必须是 0-4 之间的整数"

    labels = np.array(data_dict['label'])
    # 仅用于分层的二值标签（不改动原始 labels）
    y = (labels != 0).astype(int)

    idx_all = np.arange(n)
    idx_pos = idx_all[y == 1]
    idx_neg = idx_all[y == 0]

    rng = np.random.RandomState(seed)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    # 将正负样本均匀地分到5折（前几折会多1个，差值<=1）
    pos_splits = np.array_split(idx_pos, 5)
    neg_splits = np.array_split(idx_neg, 5)

    folds_idx = [np.concatenate([pos_splits[i], neg_splits[i]]) for i in range(5)]
    # 可选：把每折内部再打乱一下（不影响分层，只影响折内顺序）
    folds_idx = [rng.permutation(fold_idx) for fold_idx in folds_idx]

    # 选定测试折与训练折
    test_idx = folds_idx[fold]
    train_idx = np.concatenate([folds_idx[i] for i in range(5) if i != fold])

    # 构建返回字典（保持各键的一一对应）
    train_dict = {k: [v[i] for i in train_idx] for k, v in data_dict.items()}
    test_dict  = {k: [v[i] for i in test_idx]  for k, v in data_dict.items()}

    # 你也可以在这里打印每折的正样本数量以便确认
    # print("test fold pos count:", (np.array(test_dict['label']) != 0).sum())

    return train_dict, test_dict

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
    if 'data111' in args.output_dir:
        args.train_path = args.test_path
        print(f'只在111个人机实验数据集上进行五折交叉验证')
   
    train_data = pickle.load(open(args.train_path, 'rb'))
    test_data = pickle.load(open(args.test_path, 'rb'))
    if args.fold is not None:
        train, valid = split_five_fold(train_data, args.fold, seed=seed)
        train_set = ECGzheyi_two_BaseCMR(data_path=train,isTrain=True,args=args)
        valid_set = ECGzheyi_two_BaseCMR(data_path=valid,isTrain=False,args=args)
        test_set = ECGzheyi_two_BaseCMR(data_path=test_data,isTrain=False,args=args)
        print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")
    else:
        train_set = ECGzheyi_two_BaseCMR(data_path=train_data,isTrain=True,args=args)
        valid_set = ECGzheyi_two_BaseCMR(data_path=test_data,isTrain=False,args=args)
        test_set = ECGzheyi_two_BaseCMR(data_path=test_data,isTrain=False,args=args)
        print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")
   
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}")

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
    data_loader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")
    
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
        if args.cmr_pretrained_weights:
            checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')

            checkpoint_model = checkpoint['model']
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f'Load pre-trained CMR model: {msg}')
        else:
            print('Training from scratch')
        model.to(args.device)
    else:
        raise ValueError(f'Unsupported input modality: {args.input_modality}')
    
    # log
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
        
    
    
    # optimizer

    args.lr = args.blr 
    print(f'learning rate: {args.lr}')
    optimizer = get_optimizer_from_config(args, model)
    use_amp = args.use_amp
    
    
    # loss
    ClaLoss = torch.nn.BCEWithLogitsLoss()
    loss_scaler = NativeScaler()
    best_auc = float('-inf')
    BEST_PATIENCE = args.best_patience
    patient = 0

    
    if not args.only_test:
        # Start training
        misc.load_model(vars(args), model, optimizer, loss_scaler)
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(args.start_epoch, args.epochs):
            
            train_stats = train_one_epoch(model,
                                        ClaLoss,
                                        data_loader_train,
                                        optimizer,
                                        args.device,
                                        epoch,
                                        loss_scaler,
                                        log_writer,
                                        vars(args),
                                        use_amp=use_amp,
                                        args=args,
                                        )

            valid_stats, log_dict, opt_list, tgt_list =  evaluate(model,
                                    ClaLoss,
                                    data_loader_valid,
                                    args.device,
                                    log_writer,
                                    epoch,
                                    use_amp=use_amp,
                                    args=args,
                                    )
            

            test_AUC = valid_stats['AUC'].astype(float)
            curr_AUC = test_AUC
            patient += 1               
            if output_dir and curr_AUC > best_auc:
                
                best_auc = curr_AUC
                patient = 0
                misc.save_model(vars(args),
                                os.path.join(output_dir, 'best-auc.pth'),
                                epoch,
                                model,
                                optimizer,
                                loss_scaler,
                                metrics={'AUC': curr_AUC})
                
                log_dict_csv = pd.DataFrame(log_dict,index=[0])
                test_dir = os.path.join(output_dir, args.val_dir_name)
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
                log_dict_csv.to_csv(os.path.join(test_dir, f'log_dict.csv'), index=False)
                np.save(os.path.join(test_dir, f'opt_list.npy'), opt_list)
                np.save(os.path.join(test_dir, f'tgt_list.npy'), tgt_list)
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch}

            if output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + '\n\n')

            if patient > BEST_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        
        print(f'best auc: {best_auc}')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")
        
        
    elif args.only_test:
        print(f"Start testing")
        msg = model.load_state_dict(torch.load(os.path.join(output_dir, 'best-auc.pth'))['model'])
        print(f'Load pre-trained ECG model: {msg}')
        valid_stats, log_dict, opt_list, tgt_list =  evaluate(model,
                                    ClaLoss,
                                    data_loader_test,
                                    args.device,
                                    log_writer,
                                    epoch=100,
                                    use_amp=use_amp,
                                    args=args,
                                    )
        
        log_dict_csv = pd.DataFrame(log_dict,index=[0])
        test_dir = os.path.join(output_dir, args.test_dir_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        log_dict_csv.to_csv(os.path.join(test_dir, f'log_dict.csv'), index=False)
        np.save(os.path.join(test_dir, f'opt_list.npy'), opt_list)
        np.save(os.path.join(test_dir, f'tgt_list.npy'), tgt_list)
        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
