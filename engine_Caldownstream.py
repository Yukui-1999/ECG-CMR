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
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
import util.misc as misc
import util.lr_sched as lr_sched

def safe_divide(a, b, default=np.nan):
    return a / b if b != 0 else default

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
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

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
             device: torch.device,
             log_writer=None,
             epoch: int = 0,
             use_amp: bool = True,
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'valid:'
    
    tgt_list = np.array([])
    opt_list = np.array([])
    
    for sample, target in metric_logger.log_every(data_loader, 10, header):
        if args.use_gen_cmr:
            ecg = sample['ecg'].to(device, non_blocking=True).float()
            gen_cmr = sample['gen_cmr'].to(device, non_blocking=True).float()
        else:
            sample = sample.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().unsqueeze(1)

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
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).cpu().numpy()
            tgt_list = target.cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.cpu().numpy()])
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    pred_labels = (opt_list.squeeze() > 0.5).astype(int)  # 确保是1D
    true_labels = tgt_list.squeeze().astype(int)  
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sens = tp / (tp + fn)
    Spec = tn / (tn + fp)
    PPV = safe_divide(tp, tp + fp, default=0.0)   # 未预测正类时返回0
    NPV = safe_divide(tn, tn + fn, default=0.0)   # 未预测负类时返回0
    F1 = 2 * tp / (2 * tp + fp + fn)
    AUC = roc_auc_score(tgt_list, opt_list)
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f}, Acc {Acc:.3f}, Sens {Sens:.3f}, Spec {Spec:.3f}, PPV {PPV:.3f}, NPV {NPV:.3f}, F1 {F1:.3f}, AUC {AUC:.3f}'.format(
        losses=metric_logger.loss,
        Acc=Acc,
        Sens=Sens,
        Spec=Spec,
        PPV=PPV,
        NPV=NPV,
        F1=F1,
        AUC=AUC
    ))
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['Acc'] = Acc
    test_state['Sens'] = Sens
    test_state['Spec'] = Spec
    test_state['PPV'] = PPV
    test_state['NPV'] = NPV
    test_state['F1'] = F1
    test_state['AUC'] = AUC
    
    if log_writer is not None:
        epoch_1000x = int((epoch) * 1000)
        log_writer.add_scalar('valid_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('Acc', Acc, epoch_1000x)
        log_writer.add_scalar('Sens', Sens, epoch_1000x)
        log_writer.add_scalar('Spec', Spec, epoch_1000x)
        log_writer.add_scalar('PPV', PPV, epoch_1000x)
        log_writer.add_scalar('NPV', NPV, epoch_1000x)
        log_writer.add_scalar('F1', F1, epoch_1000x)
        log_writer.add_scalar('AUC', AUC, epoch_1000x)

    log_dict = {}
    log_dict['Acc'] = Acc
    log_dict['Sens'] = Sens
    log_dict['Spec'] = Spec
    log_dict['PPV'] = PPV
    log_dict['NPV'] = NPV
    log_dict['F1'] = F1
    log_dict['AUC'] = AUC
            
            
            
            
    return test_state, log_dict, opt_list, tgt_list






import numpy as np
@torch.no_grad()
def test_evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             use_amp: bool = True,
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    tgt_list = np.array([])
    opt_list = np.array([])
    
    for sample, target in metric_logger.log_every(data_loader, 10, header):
        if args.use_gen_cmr:
            ecg = sample['ecg'].to(device, non_blocking=True).float()
            gen_cmr = sample['gen_cmr'].to(device, non_blocking=True).float()
        else:
            sample = sample.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().unsqueeze(1)
        # print(f'sample shape: {sample.shape}, target shape: {target.shape}')
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
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).cpu().numpy()
            tgt_list = target.cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.cpu().numpy()])
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    pred_labels = (opt_list.squeeze() > 0.5).astype(int)  # 确保是1D
    true_labels = tgt_list.squeeze().astype(int)  
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sens = tp / (tp + fn)
    Spec = tn / (tn + fp)
    PPV = safe_divide(tp, tp + fp, default=0.0)   # 未预测正类时返回0
    NPV = safe_divide(tn, tn + fn, default=0.0)   # 未预测负类时返回0
    F1 = 2 * tp / (2 * tp + fp + fn)
    AUC = roc_auc_score(tgt_list, opt_list)
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f}, Acc {Acc:.3f}, Sens {Sens:.3f}, Spec {Spec:.3f}, PPV {PPV:.3f}, NPV {NPV:.3f}, F1 {F1:.3f}, AUC {AUC:.3f}'.format(
        losses=metric_logger.loss,
        Acc=Acc,
        Sens=Sens,
        Spec=Spec,
        PPV=PPV,
        NPV=NPV,
        F1=F1,
        AUC=AUC
    ))
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['Acc'] = Acc
    test_state['Sens'] = Sens
    test_state['Spec'] = Spec
    test_state['PPV'] = PPV
    test_state['NPV'] = NPV
    test_state['F1'] = F1
    test_state['AUC'] = AUC
    
    if log_writer is not None:
        epoch_1000x = 0
        log_writer.add_scalar('test_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('test_Acc', Acc, epoch_1000x)
        log_writer.add_scalar('test_Sens', Sens, epoch_1000x)
        log_writer.add_scalar('test_Spec', Spec, epoch_1000x)
        log_writer.add_scalar('test_PPV', PPV, epoch_1000x)
        log_writer.add_scalar('test_NPV', NPV, epoch_1000x)
        log_writer.add_scalar('test_F1', F1, epoch_1000x)
        log_writer.add_scalar('test_AUC', AUC, epoch_1000x)

    log_dict = {}
    log_dict['Acc'] = Acc
    log_dict['Sens'] = Sens
    log_dict['Spec'] = Spec
    log_dict['PPV'] = PPV
    log_dict['NPV'] = NPV
    log_dict['F1'] = F1
    log_dict['AUC'] = AUC
            
            
            
            
    return test_state, log_dict, opt_list, tgt_list
