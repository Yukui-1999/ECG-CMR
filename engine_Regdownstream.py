# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch
import torchmetrics
import pandas as pd
import util.misc as misc
import util.lr_sched as lr_sched
from scipy.stats import pearsonr

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    args=None,
                    ) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config.get('accum_iter', 1)
    max_norm = config.get('max_norm', None)

    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        if args.use_gen_cmr:
            ecg = samples['ecg'].to(device, non_blocking=True).float()
            gen_cmr = samples['gen_cmr'].to(device, non_blocking=True).float()
        else:
            samples = samples.to(device, non_blocking=True).float()
        
        targets = targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.use_gen_cmr:
                outputs = model(ecg, gen_cmr)
            else:
                outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import numpy as np
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             train_data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             epoch: int = 0,
             use_amp: bool = True,
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 
                 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 
                 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 
                 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 
                 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 
                 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 
                 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 
                 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 
                 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 
                 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 
                 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 
                 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 
                 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 
                 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 
                 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 
                 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    cor_index = [i.replace(' ', '_') for i in cor_index]
    tgt_list = np.array([])
    opt_list = np.array([])
    
    for sample, target in metric_logger.log_every(data_loader, 10, header):
        
        if args.use_gen_cmr:
            ecg = sample['ecg'].to(device, non_blocking=True).float()
            gen_cmr = sample['gen_cmr'].to(device, non_blocking=True).float()
        else:
            sample = sample.to(device, non_blocking=True)
        
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.use_gen_cmr:
                if ecg.ndim == 4:
                    logits_list = []
                    for i in range(ecg.size(1)):
                        logits = model(ecg[:, i], gen_cmr)
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
                else:
                    output = model(ecg, gen_cmr)
            else:
                if sample.ndim == 4 and not 'CMRmode' in args.output_dir:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
                else:
                    output = model(sample)
            loss = criterion(output, target)
        
            output = train_data_loader.dataset.scaler.inverse_transform(output.cpu().numpy())
            target = train_data_loader.dataset.scaler.inverse_transform(target.cpu().numpy())
            if len(opt_list) == 0:
                opt_list = output
                tgt_list = target
            else:
                opt_list = np.concatenate([opt_list, output], axis=0)
                tgt_list = np.concatenate([tgt_list, target], axis=0)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        metric_logger.update(loss=loss_value)
        
    mae = np.mean(np.abs(opt_list - tgt_list), axis=0)
    rmse = np.sqrt(np.mean((opt_list - tgt_list) ** 2, axis=0))
    r_squared = 1 - np.sum((opt_list - tgt_list) ** 2, axis=0) / np.sum((tgt_list - np.mean(tgt_list, axis=0)) ** 2, axis=0)
    corr_list = []
    p_value_list = []
    for i in range(82):
        corr, p_value = pearsonr(opt_list[:, i].flatten(), tgt_list[:, i].flatten())
        p_value_list.append(p_value)
        corr_list.append(corr)  
    
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f} mean_mae {mean_mae:.3f} mean_r2 {mean_r2:.3f}, mean_r {mean_r:.3f}, mean_rmse {mean_rmse:.3f}'
          .format(losses=metric_logger.loss,
                mean_mae=mae.mean(),
                mean_r2=r_squared.mean(),
                mean_r=np.mean(corr_list),
                mean_rmse=rmse.mean()))
        
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['mean_mae'] = np.mean(mae)
    test_state['mean_r2'] = np.mean(r_squared)
    test_state['mean_r'] = np.mean(corr_list)
    test_state['mean_rmse'] = np.mean(rmse)
    for index, cor in enumerate(cor_index):
        if index > 17:
            continue
        mae_name = cor+'_mae'
        r2_name = cor+'_r2'
        r_name = cor+'_r'
        rmse_name = cor+'_rmse'
        
        test_state[mae_name] = mae[index]
        test_state[rmse_name] = rmse[index]
        test_state[r2_name] = r_squared[index]
        test_state[r_name] = corr_list[index]

    if log_writer is not None:
        epoch_1000x = int((epoch) * 1000)
        log_writer.add_scalar('test_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('mean_mae', test_state['mean_mae'], epoch_1000x)
        log_writer.add_scalar('mean_r2', test_state['mean_r2'], epoch_1000x)
        log_writer.add_scalar('mean_r', test_state['mean_r'], epoch_1000x)
        log_writer.add_scalar('mean_rmse', test_state['mean_rmse'], epoch_1000x)
        for index, cor in enumerate(cor_index):
            if index > 17:
                continue
            mae_name = cor+'_mae'
            r2_name = cor+'_r2'
            r_name = cor+'_r'
            rmse_name = cor+'_rmse'

            log_writer.add_scalar(mae_name, test_state[mae_name], epoch_1000x)
            log_writer.add_scalar(rmse_name, test_state[rmse_name], epoch_1000x)
            log_writer.add_scalar(r2_name, test_state[r2_name], epoch_1000x)
            log_writer.add_scalar(r_name, test_state[r_name], epoch_1000x)
    
    
    log_dict = {}
    log_dict['mean_mae'] = test_state['mean_mae']
    log_dict['mean_r2'] = test_state['mean_r2']
    log_dict['mean_r'] = test_state['mean_r']
    log_dict['mean_rmse'] = test_state['mean_rmse']
    for index, cor in enumerate(cor_index):
        mae_name = cor+'_mae'
        r2_name = cor+'_r2'
        r_name = cor+'_r'
        r_p_name = cor+'_p'
        rmse_name = cor+'_rmse'
        log_dict[mae_name] = mae[index]
        log_dict[r2_name] = r_squared[index]
        log_dict[r_name] = corr_list[index]
        log_dict[r_p_name] = p_value_list[index]
        log_dict[rmse_name] = rmse[index]

    
    return test_state, log_dict, opt_list, tgt_list
