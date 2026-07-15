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

from engine_Caldownstream import train_one_epoch, evaluate
from models import encoder
from CMR_encoder import models_vit
from util.losses import ClipLoss
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import ECGBaseDis, CMRBaseDis, ECGDis_wGenCMR
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
    parser.add_argument('--fold', default=0, type=int, help='fold 0,1,2,3,4')
    parser.add_argument('--only_test', default=False, type=bool, help='only test')
    return parser

def extact_info(eid_list,csv_data,work_edu_csv,save_name):
    select_index = [
        '4079-2.0',
        '4080-2.0',
        '21003-2.0',
        '31-0.0',
        '21000-0.0',
        '21001-2.0',
        '20160-2.0',
        '1160-2.0',
        '1558-2.0',
        '6142-2.0',
        '6138-2.0',
        '24100-2.0',
        '24101-2.0',
        '24102-2.0',
        '24103-2.0',
        '24104-2.0',
        '24105-2.0',
        '24106-2.0',
        '24107-2.0',    
        '24108-2.0',
        '24109-2.0',
    ]
    
    select_index_name_number = {
        'DBP':[],
        'SBP':[],
        'Age':[],
        'Sex_Female':0,
        'Sex_Male':0,
        'Ethnic_White':0,
        'Ethnic_Mixed':0,
        'Ethnic_Asian or Asian British':0,
        'Ethnic_Black or Black British':0,
        'Ethnic_Chinese':0,
        'Ethnic_Other ethnic group':0,
        'Ethnic_Unknown':0,
        'BMI':[],
        'Ever_smoked_yes':0,
        'Ever_smoked_no':0,
        'Ever_smoked_Unknown':0,
        'Sleep_duration':[],
        'Alcohol_Daily or almost daily':0,
        'Alcohol_Three or four times a week':0,
        'Alcohol_Once or twice a week':0,
        'Alcohol_One to three times a month':0,
        'Alcohol_Special occasions only':0,
        'Alcohol_Never':0,
        'Alcohol_Unknown':0,
        'Work_In paid employment or self-employed':0,
        'Work_Retired':0,
        'Work_Looking after home and/or family':0,
        'Work_Unable to work because of sickness or disability':0,
        'Work_Unemployed':0,
        'Work_Doing unpaid or voluntary work':0,
        'Work_Full or part-time student':0,
        'Work_Unknown':0,
        'Education_College or University degree':0,
        'Education_A levels/AS levels or equivalent':0,
        'Education_O levels/GCSEs or equivalent':0,
        'Education_CSEs or equivalent':0,
        'Education_NVQ or HND or HNC or equivalent':0,
        'Education_Other professional qualifications eg: nursing, teaching':0,
        'Education_Unknown':0,
        'LV end diastolic volume':[],
        'LV end systolic volume':[],
        'LV stroke volume':[],
        'LV ejection fraction':[],
        'LV cardiac output':[],
        'LV myocardial mass':[],
        'RV end diastolic volume':[],
        'RV end systolic volume':[],
        'RV stroke volume':[],
        'RV ejection fraction':[],
    }
    
    
    for eid in tqdm(eid_list):
        if int(eid) not in csv_data['eid'].values:
            print(f'{eid} not in csv_data')
            continue
        if int(eid) not in work_edu_csv['eid'].values:
            print(f'{eid} not in work_edu_csv')
            continue
        index = np.where(csv_data['eid'].values == int(eid))[0][0]
        index_work_edu = np.where(work_edu_csv['eid'].values == int(eid))[0][0]
        for i, index_name in enumerate(select_index):
            if index_name == '6142-2.0' or index_name == '6138-2.0':
                work_edu_row = work_edu_csv.iloc[index_work_edu]
                
                if index_name == '6142-2.0':
                    work = work_edu_row['6142-2.0']
                    if pd.isna(work):
                        select_index_name_number['Work_Unknown'] += 1
                    else:
                        work = int(work)
                        if work == 1:
                            select_index_name_number['Work_In paid employment or self-employed'] += 1
                        elif work == 2:
                            select_index_name_number['Work_Retired'] += 1
                        elif work == 3:
                            select_index_name_number['Work_Looking after home and/or family'] += 1
                        elif work == 4:
                            select_index_name_number['Work_Unable to work because of sickness or disability'] += 1
                        elif work == 5:
                            select_index_name_number['Work_Unemployed'] += 1
                        elif work == 6:
                            select_index_name_number['Work_Doing unpaid or voluntary work'] += 1
                        elif work == 7:
                            select_index_name_number['Work_Full or part-time student'] += 1
                        else:
                            select_index_name_number['Work_Unknown'] += 1
                
                elif index_name == '6138-2.0':
                    education = work_edu_row['6138-2.0']
                    if pd.isna(education):
                        select_index_name_number['Education_Unknown'] += 1
                    else:
                        education = int(education)
                        if education == 1:
                            select_index_name_number['Education_College or University degree'] += 1
                        elif education == 2:
                            select_index_name_number['Education_A levels/AS levels or equivalent'] += 1
                        elif education == 3:
                            select_index_name_number['Education_O levels/GCSEs or equivalent'] += 1
                        elif education == 4:
                            select_index_name_number['Education_CSEs or equivalent'] += 1
                        elif education == 5:
                            select_index_name_number['Education_NVQ or HND or HNC or equivalent'] += 1
                        elif education == 6:
                            select_index_name_number['Education_Other professional qualifications eg: nursing, teaching'] += 1
                        else:
                            select_index_name_number['Education_Unknown'] += 1
            
            else:
                data_row = csv_data.iloc[index]
                if index_name == '4079-2.0':
                    select_index_name_number['DBP'].append(float(data_row[index_name]))
                elif index_name == '4080-2.0':
                    select_index_name_number['SBP'].append(float(data_row[index_name]))
                elif index_name == '21003-2.0':
                    select_index_name_number['Age'].append(float(data_row[index_name]))
                elif index_name == '31-0.0':
                    sex = data_row[index_name]
                    sex = int(sex)
                    if sex == 0:
                        select_index_name_number['Sex_Female'] += 1
                    else:
                        select_index_name_number['Sex_Male'] += 1
                elif index_name == '21000-0.0':
                    ethnic = data_row[index_name]
                    if pd.isna(ethnic):
                        select_index_name_number['Ethnic_Unknown'] += 1
                    else:
                        ethnic = str(ethnic)
                        if ethnic.startswith('1'):
                            select_index_name_number['Ethnic_White'] += 1
                        elif ethnic.startswith('2'):
                            select_index_name_number['Ethnic_Mixed'] += 1
                        elif ethnic.startswith('3'):
                            select_index_name_number['Ethnic_Asian or Asian British'] += 1
                        elif ethnic.startswith('4'):
                            select_index_name_number['Ethnic_Black or Black British'] += 1
                        elif ethnic.startswith('5'):
                            select_index_name_number['Ethnic_Chinese'] += 1
                        elif ethnic.startswith('6'):
                            select_index_name_number['Ethnic_Other ethnic group'] += 1
                        else:
                            select_index_name_number['Ethnic_Unknown'] += 1
                elif index_name == '21001-2.0':
                    select_index_name_number['BMI'].append(float(data_row[index_name]))
                elif index_name == '20160-2.0':
                    smoke = data_row[index_name]
                    if pd.isna(smoke):
                        select_index_name_number['Ever_smoked_Unknown'] += 1
                    else:
                        smoke = int(smoke)
                        if smoke == 1:
                            select_index_name_number['Ever_smoked_yes'] += 1
                        else:
                            select_index_name_number['Ever_smoked_no'] += 1
                elif index_name == '1160-2.0':
                    select_index_name_number['Sleep_duration'].append(float(data_row[index_name]))
                elif index_name == '1558-2.0':
                    drike = data_row[index_name]
                    if pd.isna(drike):
                        select_index_name_number['Alcohol_Unknown'] += 1
                    else:
                        drike = int(drike)
                        if drike == 1:
                            select_index_name_number['Alcohol_Daily or almost daily'] += 1
                        elif drike == 2:
                            select_index_name_number['Alcohol_Three or four times a week'] += 1
                        elif drike == 3:
                            select_index_name_number['Alcohol_Once or twice a week'] += 1
                        elif drike == 4:
                            select_index_name_number['Alcohol_One to three times a month'] += 1
                        elif drike == 5:
                            select_index_name_number['Alcohol_Special occasions only'] += 1
                        elif drike == 6:
                            select_index_name_number['Alcohol_Never'] += 1
                        else:
                            select_index_name_number['Alcohol_Unknown'] += 1
                elif index_name == '24100-2.0':
                    select_index_name_number['LV end diastolic volume'].append(float(data_row[index_name]))
                elif index_name == '24101-2.0':
                    select_index_name_number['LV end systolic volume'].append(float(data_row[index_name]))
                elif index_name == '24102-2.0':
                    select_index_name_number['LV stroke volume'].append(float(data_row[index_name]))
                elif index_name == '24103-2.0':
                    select_index_name_number['LV ejection fraction'].append(float(data_row[index_name]))
                elif index_name == '24104-2.0':
                    select_index_name_number['LV cardiac output'].append(float(data_row[index_name]))
                elif index_name == '24105-2.0': 
                    select_index_name_number['LV myocardial mass'].append(float(data_row[index_name]))
                elif index_name == '24106-2.0':
                    select_index_name_number['RV end diastolic volume'].append(float(data_row[index_name]))
                elif index_name == '24107-2.0':
                    select_index_name_number['RV end systolic volume'].append(float(data_row[index_name]))
                elif index_name == '24108-2.0':
                    select_index_name_number['RV stroke volume'].append(float(data_row[index_name]))
                elif index_name == '24109-2.0':
                    select_index_name_number['RV ejection fraction'].append(float(data_row[index_name]))
                else:
                    raise ValueError(f'Unknown index name: {index_name}')
    
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



def cross_validation_split(eid, k, fold):
    

    n = len(eid)
    k = 5
    split_size = n // k  
    remainder = n % k   
    split_eid = []
    start = 0
    for i in range(k):
        end = start + split_size + (1 if i < remainder else 0)
        split_eid.append(eid[start:end])
        start = end
    # 检查结果
    for i, part in enumerate(split_eid):
        print(f"Part {i+1}: {len(part)} samples, eid[0]:{part[0][0]}, eid[-1]:{part[-1][0]}")
        
    train = []
    valid = []
    for i in range(k):
        if i == fold:
            valid = split_eid[i]
        else:
            train += split_eid[i]
    print(f"Train size: {len(train)}, valid size: {len(valid)}")
    return train, valid
        
def set_true_false_eid(eid,dis,health_magnification):
    health_eid = json.load(open('/home/liziyu/CMR/data/cmr_files_wo_I.json', "r"))
    print(f'health_eid: {len(health_eid)}')
    eid = [i.split('/')[-1].split('_')[0] for i in eid]
    
    if dis == 'cad':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/cad_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    elif dis == 'cm':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/xjb_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    elif dis == 'hf':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/xs_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    
    
    dis_eid = sorted(set(dis_eid) & set(eid)) # sorted is important 因为不受种子控制
    health_eid = sorted(set(health_eid) & set(eid))
    
    
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
    

    final_eid = dis_eid + health_eid
    label = [1] * len(dis_eid) + [0] * len(health_eid)
    final_eid = list(zip(final_eid, label))
    random.shuffle(final_eid)

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
    if args.input_modality == 'ECG':
        eid_json = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/all_data_v1.json"
        eid = json.load(open(eid_json, 'r'))
        k = 5
        mix_eid = set_true_false_eid(eid,args.dis,args.health_magnification)

        train, valid = cross_validation_split(mix_eid, k, args.fold)
        if args.use_gen_cmr:   
            train_set = ECGDis_wGenCMR(data=train,isTrain=True)
            valid_set = ECGDis_wGenCMR(data=valid,isTrain=False)
        else:
            train_set = ECGBaseDis(data=train,isTrain=True)
            valid_set = ECGBaseDis(data=valid,isTrain=False)
        
    elif args.input_modality == 'CMR':
        eid_json = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/all_data_v1.json"
        eid = json.load(open(eid_json, 'r'))
        k = 5
        mix_eid = set_true_false_eid(eid,args.dis,args.health_magnification)
        
        train, valid = cross_validation_split(mix_eid, k, args.fold)        
        train_set = CMRBaseDis(data=train)
        valid_set = CMRBaseDis(data=valid)
    else:
        raise ValueError(f'Unsupported input modality: {args.input_modality}')
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
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}")
    
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
            print(f'No pre-trained CMR model, training from scratch')
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
                test_dir = os.path.join(output_dir, args.test_dir_name)
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
                                    data_loader_valid,
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
