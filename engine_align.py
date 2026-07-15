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

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(ECG_model: torch.nn.Module,
                    CMR_model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    ) -> Dict[str, float]:
    ECG_model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config.get('accum_iter', 1)
    max_norm = config.get('max_norm', None)

    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (ecg, cmr) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        ecg = ecg.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)
        # print(f"ecg shape: {ecg.shape}, cmr shape: {cmr.shape}")
        # print(f'ecg dtype: {ecg.dtype}, cmr dtype: {cmr.dtype}')
        # print(f'ecg.min: {ecg.min()}, ecg.max: {ecg.max()}')
        # print(f'cmr.min: {cmr.min()}, cmr.max: {cmr.max()}')
        # print(f'ecg.mean: {ecg.mean()}, ecg.std: {ecg.std()}')
        # print(f'cmr.mean: {cmr.mean()}, cmr.std: {cmr.std()}')
        # exit()
        with torch.cuda.amp.autocast(enabled=use_amp):
            ecg_outputs = ECG_model(ecg)
            cmr_outputs = CMR_model(cmr)
            loss = criterion(ecg_outputs, cmr_outputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=ECG_model.parameters(),
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
def evaluate(ECG_model: torch.nn.Module,
             CMR_model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             epoch: int,
             log_writer=None,
             config: Optional[dict] = None,
             use_amp: bool = True,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    ECG_model.eval()
    CMR_model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    accum_iter = config.get('accum_iter', 1)
    
    val_L2 = 0
    val_cos = 0
    result_list = []
    label_list = []
    for data_iter_step, (ecg, cmr) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        ecg = ecg.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if ecg.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(ecg.size(1)):
                    logits = ECG_model(ecg[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                ecg_outputs = logits_list.mean(dim=1)
                cmr_outputs = CMR_model(cmr)
                
                # global loss
                ecg_embed = torch.nn.functional.normalize(ecg_outputs, dim=-1)
                features = torch.nn.functional.normalize(cmr_outputs, dim=-1)
                L2 = np.mean(np.linalg.norm(ecg_embed.squeeze().cpu().numpy() - features.squeeze().cpu().numpy(), ord=2, axis=1))
                cosin_sim = np.mean(np.sum(ecg_embed.squeeze().cpu().numpy() * features.squeeze().cpu().numpy(), axis=1) / (np.linalg.norm(ecg_embed.squeeze().cpu().numpy(), ord=2, axis=1) * np.linalg.norm(features.squeeze().cpu().numpy(), ord=2, axis=1)))
                val_L2 += L2
                val_cos += cosin_sim
                
                prob = ecg_embed @ features.T
                prob = torch.nn.functional.softmax(prob, dim=-1)
                result = torch.argmax(prob, dim=-1)
                result_list += result.cpu().numpy().tolist()
                label_list += [i for i in range (len(ecg_outputs))]

            
            else:
                ecg_outputs = ECG_model(ecg)
                cmr_outputs = CMR_model(cmr)
            loss = criterion(ecg_outputs, cmr_outputs)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training in val stage")
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('test_loss', loss_value_reduce, epoch_1000x)
    
    right_num = 0
    for i in range(len(result_list)):
        if result_list[i] == label_list[i]:
            right_num += 1
    acc = right_num / len(result_list)
    print(f'EPOCH: {epoch}, Accuracy: {acc:.4f}')
    print('L2:', val_L2/len(data_loader))
    print('sim:', val_cos/len(data_loader))  
         
    log_writer.add_scalar('Accuracy', acc, epoch_1000x)
    log_writer.add_scalar('L2', val_L2/len(data_loader), epoch_1000x)
    log_writer.add_scalar('cosin_sim', val_cos/len(data_loader), epoch_1000x)
    
    
    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("validation stats:", metric_logger)
   

    return valid_stats
