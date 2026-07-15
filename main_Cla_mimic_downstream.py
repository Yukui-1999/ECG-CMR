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

from engine_Caldownstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGBaseMIMICDis, ECGMIMICDis_wGenCMR
from util.val_result import process_val_result
from ECG_genCMR_model import ECG_genCMR_model
def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
    
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/configs/Cla/st_mem_align.yaml', type=str, help='ecg config path')
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default='/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth', type=str, help='pretrained weights path')
    parser.add_argument('--use_gen_cmr', default=False, type=bool, help='use generated cmr')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/', type=str, help='number of classes')
    parser.add_argument('--test_dir_name', default='test_debug', type=str, help='test dir name')
    
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
    parser.add_argument('--train_percent', default=1, type=float, help='0.1,0.25,0.5,0.75,1')
    parser.add_argument('--only_test', default=False, type=bool, help='only test')
    return parser


def extact_info(eid_list,base_df,admission_df,save_name):
    eid_list = [i.replace('/mnt/sda1/lihaitao/datasets/ECG/','') for i in eid_list]
    eid_list = [i.replace('.dat','') for i in eid_list]

    type_id = [
        'gender',
        'race',
        'marital_status',
        'age',
        ]
    
    select_index_name_number = {
        'Age':[],
        'Sex_Female':0,
        'Sex_Male':0,
        'Sex_Unknown':0,
        'marital_status_WIDOWED':0,
        'marital_status_SINGLE':0,
        'marital_status_MARRIED':0,
        'marital_status_DIVORCED':0,
        'marital_status_Unknown':0,
        'Ethnic_WHITE':0,
        'Ethnic_BLACK':0,
        'Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':0,
        'Ethnic_PORTUGUESE':0,
        'Ethnic_HISPANIC':0,
        'Ethnic_ASIAN':0,
        'Ethnic_SOUTH AMERICAN':0,
        'Ethnic_AMERICAN INDIAN':0,
        'Ethnic_MULTIPLE RACE':0,
        'Ethnic_Unknown':0,
    }
    
    
    for eid in tqdm(eid_list):
        if int(eid.split('/')[-3][1:]) not in admission_df['subject_id'].values:
            # print(f"subject_id {eid} not in admission_df")
            select_index_name_number['Ethnic_Unknown'] += 1
            select_index_name_number['marital_status_Unknown'] += 1
            index_base = np.where(base_df['file_name'].values == eid)[0][0]
            base_row = base_df.iloc[index_base]

            gender = base_row['gender']
            if pd.isna(gender):
                select_index_name_number['Sex_Unknown'] += 1
            else:
                if gender == 'M':
                    select_index_name_number['Sex_Male'] += 1
                elif gender == 'F':
                    select_index_name_number['Sex_Female'] += 1
                else:
                    select_index_name_number['Sex_Unknown'] += 1

            age = base_row['age']
            select_index_name_number['Age'].append(float(age))
            continue
        
        index_base = np.where(base_df['file_name'].values == eid)[0][0]
        index_admission = np.where(admission_df['subject_id'].values == int(eid.split('/')[-3][1:]))[0][0]
        
        
        for i, index_name in enumerate(type_id):
            if index_name == 'gender' or index_name == 'age':
                base_row = base_df.iloc[index_base]
                if index_name == 'gender':
                    gender = base_row['gender']
                    if pd.isna(gender):
                        select_index_name_number['Sex_Unknown'] += 1
                    else:
                        if gender == 'M':
                            select_index_name_number['Sex_Male'] += 1
                        elif gender == 'F':
                            select_index_name_number['Sex_Female'] += 1
                        else:
                            select_index_name_number['Sex_Unknown'] += 1
                elif index_name == 'age':
                    age = base_row['age']
                    select_index_name_number['Age'].append(float(age))
            
            elif index_name == 'marital_status' or index_name == 'race':
                admission_df_row = admission_df.iloc[index_admission]
                if index_name == 'marital_status':
                    marital_status = admission_df_row['marital_status']
                    if pd.isna(marital_status):
                        select_index_name_number['marital_status_Unknown'] += 1
                    else:
                        if marital_status == 'WIDOWED':
                            select_index_name_number['marital_status_WIDOWED'] += 1
                        elif marital_status == 'SINGLE':
                            select_index_name_number['marital_status_SINGLE'] += 1
                        elif marital_status == 'MARRIED':
                            select_index_name_number['marital_status_MARRIED'] += 1
                        elif marital_status == 'DIVORCED':
                            select_index_name_number['marital_status_DIVORCED'] += 1
                        else:
                            select_index_name_number['marital_status_Unknown'] += 1
                elif index_name == 'race':
                    race = admission_df_row['race']
                    if pd.isna(race):
                        select_index_name_number['Ethnic_Unknown'] += 1
                    else:
                        if race.startswith('WHITE'):
                            select_index_name_number['Ethnic_WHITE'] += 1
                        elif race.startswith('BLACK'):
                            select_index_name_number['Ethnic_BLACK'] += 1
                        elif race.startswith('NATIVE HAWAIIAN'):
                            select_index_name_number['Ethnic_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'] += 1
                        elif race.startswith('PORTUGUESE'):
                            select_index_name_number['Ethnic_PORTUGUESE'] += 1
                        elif race.startswith('HISPANIC'):
                            select_index_name_number['Ethnic_HISPANIC'] += 1
                        elif race.startswith('SOUTH AMERICAN'):
                            select_index_name_number['Ethnic_SOUTH AMERICAN'] += 1
                        elif race.startswith('AMERICAN INDIAN'):
                            select_index_name_number['Ethnic_AMERICAN INDIAN'] += 1
                        elif race.startswith('MULTIPLE RACE'):
                            select_index_name_number['Ethnic_MULTIPLE RACE'] += 1 
                        else:
                            select_index_name_number['Ethnic_Unknown'] += 1
                
            
    
    processed_dict = {}
    for key, value in select_index_name_number.items():
        if isinstance(value, list) and len(value) > 0:
            mean = np.nanmean(value)
            std = np.nanstd(value)
            processed_dict[key] = f"{mean:.3f}±{std:.3f}"
        else:
            processed_dict[key] = value

    # 转换为DataFrame并导出CSV
    df = pd.DataFrame([processed_dict])
    df.to_csv(save_name, index=False)

    print(f"处理完成，结果已保存到 {save_name}")
    
    


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



def train_val_test_split(eid, train=0.6, valid=0.2, test=0.2):
    
    assert train + valid + test == 1, f"train + valid + test must be 1, but got {train} + {valid} + {test}"
    assert len(eid) > 0, f"eid list is empty"
    
    train_size = int(len(eid) * train)
    valid_size = int(len(eid) * valid)

    train = eid[:train_size]
    valid = eid[train_size:train_size + valid_size]
    test = eid[train_size + valid_size:]
    print('train[0]:', train[0])
    print('valid[0]:', valid[0])
    print('test[0]:', test[0])
    print(f"Train size: {len(train)}, valid size: {len(valid)}, test size: {len(test)}")
    
    return train, valid, test
        
def set_true_false_eid(args,dis,health_magnification):
    # health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_woI.json', "r"))
    # print(f'health_eid: {len(health_eid)}')
    
    if dis == 'cad':
        
        dis_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_CAD.json', "r"))
        health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_CAD.json', "r"))
        
            
        print('health_eid load from wo_CAD: /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_CAD.json')
        print(f'health_eid: {len(health_eid)}')
    elif dis == 'cm':
        
        dis_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_CM.json', "r"))
        health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_CM.json', "r"))
        
        print('health_eid load from wo_CM: /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_CM.json')
        print(f'health_eid: {len(health_eid)}')
    elif dis == 'hf':
        
        dis_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_HF.json', "r"))
        health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_HF.json', "r"))
        
        print('health_eid load from wo_HF: /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_HF.json')
        print(f'health_eid: {len(health_eid)}')
    elif dis == 'ph':
        
        dis_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_PH.json', "r"))
        health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_PH.json', "r"))
        
        print('health_eid load from wo_PH: /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_PH.json')
        print(f'health_eid: {len(health_eid)}')
    elif dis == 'pc':
        
        dis_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_PC.json', "r"))
        health_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_PC.json', "r"))
        
        print('health_eid load from wo_PC: /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_wo_PC.json')
        print(f'health_eid: {len(health_eid)}')
    
    assert len(set(dis_eid) & set(health_eid)) == 0, f"Dis EID and Health EID overlap: {len(set(dis_eid) & set(health_eid))}"
    
    if health_magnification > 0:
        max_size = min(len(dis_eid) * health_magnification, len(health_eid) )
        if max_size == len(health_eid):
            print(f'----------------------------------------------')
            print(f"Health EID size * health_magnification is greater than the original size")
            print(f'----------------------------------------------')
        health_eid = np.random.choice(health_eid, size=max_size, replace=False).tolist()
    print(f'----------------------------------------------')
    print(f"Dis {dis}: {len(dis_eid)}, Health EID: {len(health_eid)}")
    print(f'----------------------------------------------')
    # extact_info(dis_eid,pd.read_csv("/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/records_w_diag_icd10.csv", low_memory=False),
    #             pd.read_csv('/mnt/data2/MIMIC4_3.1_version/mimiciv/3.1/hosp/admissions.csv', low_memory=False),
    #             f'/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/mimic/{dis}_positive.csv')
    # extact_info(health_eid,pd.read_csv("/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/records_w_diag_icd10.csv", low_memory=False),
    #             pd.read_csv('/mnt/data2/MIMIC4_3.1_version/mimiciv/3.1/hosp/admissions.csv', low_memory=False),
    #             f'/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/mimic/{dis}_negative.csv')
    final_eid = dis_eid + health_eid
    # extact_info(final_eid,pd.read_csv("/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/records_w_diag_icd10.csv", low_memory=False),
    #             pd.read_csv('/mnt/data2/MIMIC4_3.1_version/mimiciv/3.1/hosp/admissions.csv', low_memory=False),
    #             f'/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/mimic/{dis}_positive_negative.csv')
    label = [1] * len(dis_eid) + [0] * len(health_eid)
    final_eid = list(zip(final_eid, label))
    # with open(os.path.join(args.output_dir, f'{dis}_final_eid.json'), 'w') as f:
    #     json.dump(final_eid, f)
    # exit(0)
    random.shuffle(final_eid)
    print(f'final_eid[0]: {final_eid[0]}')
    print(f'final_eid[-1]: {final_eid[-1]}')
    return final_eid

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
    
    
    mix_eid = set_true_false_eid(args, args.dis, args.health_magnification)
    # exit()
    train, valid, test = train_val_test_split(mix_eid, train=0.6, valid=0.2, test=0.2)
    if args.use_gen_cmr:   
        train_set = ECGMIMICDis_wGenCMR(data=train,train_percent=args.train_percent,isTrain=True)
        valid_set = ECGMIMICDis_wGenCMR(data=valid,isTrain=False)
        test_set = ECGMIMICDis_wGenCMR(data=test,isTrain=False)
    else:
        train_set = ECGBaseMIMICDis(data=train,train_percent=args.train_percent,isTrain=True)
        valid_set = ECGBaseMIMICDis(data=valid,isTrain=False)
        test_set = ECGBaseMIMICDis(data=test,isTrain=False)
        
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")

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
    

    
    # ECG model input shape (batchsize, 12, 2250)
    if args.use_gen_cmr:
            model = ECG_genCMR_model(args)
            model.to(args.device)
    else:
        model = build_ecg_model(args.ecg_config_path)
        model.to(args.device)
    
    
    # log
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
        
    
    
    # optimizer

    args.lr = args.blr 
    print(f'learning rate: {args.lr}')
    optimizer = get_optimizer_from_config(args, model)
    
    
    
    # loss
    ClaLoss = torch.nn.BCEWithLogitsLoss()
    loss_scaler = NativeScaler()
    best_auc = float('-inf')
    BEST_PATIENCE = args.best_patience
    patient = 0
    best_model = None

    
    if not args.only_test:
        # Start training
        misc.load_model(vars(args), model, optimizer, loss_scaler)
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        use_amp = args.use_amp
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
            

            valid_AUC = valid_stats['AUC'].astype(float)
            curr_AUC = valid_AUC
            patient += 1               
            if output_dir and curr_AUC > best_auc:
                
                best_model = model
                best_auc = curr_AUC
                patient = 0
                misc.save_model(vars(args),
                                os.path.join(output_dir, 'best-auc.pth'),
                                epoch,
                                model,
                                optimizer,
                                loss_scaler,
                                metrics={'AUC': curr_AUC})
                
                # misc.save_model(vars(args),
                #                 os.path.join(output_dir, f'best-auc-{curr_AUC}.pth'),
                #                 epoch,
                #                 model,
                #                 optimizer,
                #                 loss_scaler,
                #                 metrics={'AUC': curr_AUC})
        
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

        
        print(f'begin testing')
        msg = model.load_state_dict(torch.load(os.path.join(output_dir, 'best-auc.pth'))['model'])
        print(f'load best auc model: {msg}')
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
                
                
    elif args.only_test:
        print(f'begin testing')
        print(f'load {os.path.join(output_dir, "best-auc.pth")}')
        msg = model.load_state_dict(torch.load(os.path.join(output_dir, 'best-auc.pth'))['model'])
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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
