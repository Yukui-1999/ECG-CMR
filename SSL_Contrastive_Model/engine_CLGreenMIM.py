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
import math
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    loss_fn = torch.nn.MSELoss()
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        
        
        cmr = batch['cmr'].float().to(device, non_blocking=True)
        la_cmr = batch['la_cmr'].float().to(device, non_blocking=True)
        

        with torch.cuda.amp.autocast():
            loss_dict = model(cmr, la_cmr,args.mask_ratio)
            loss1 = loss_dict['loss1']
            loss2 = loss_dict['loss2']
            cliploss = loss_dict['cliploss']
            loss = loss1 + loss2 + cliploss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss1=loss1.item())
        metric_logger.update(loss2=loss2.item())
        metric_logger.update(cliploss=cliploss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss1_reduce = misc.all_reduce_mean(loss1.item())
        loss2_reduce = misc.all_reduce_mean(loss2.item())
        cliploss_reduce = misc.all_reduce_mean(cliploss.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('loss1', loss1_reduce, epoch_1000x)
            log_writer.add_scalar('loss2', loss2_reduce, epoch_1000x)
            log_writer.add_scalar('cliploss', cliploss_reduce, epoch_1000x)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



from sklearn.metrics import roc_auc_score
@torch.no_grad()
def evaluate(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    loss_fn = torch.nn.MSELoss()
    output = []
    label = []
    accum_iter = args.accum_iter
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

       
        cmr = batch['cmr'].float().to(device, non_blocking=True)
        la_cmr = batch['la_cmr'].float().to(device, non_blocking=True)
        

        with torch.cuda.amp.autocast():
            loss_dict = model(cmr, la_cmr,args.mask_ratio)
            loss1 = loss_dict['loss1']
            loss2 = loss_dict['loss2']
            cliploss = loss_dict['cliploss']
            loss = loss1 + loss2 + cliploss
        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss1=loss1.item())
        metric_logger.update(loss2=loss2.item())
        metric_logger.update(cliploss=cliploss.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss1_reduce = misc.all_reduce_mean(loss1.item())
        loss2_reduce = misc.all_reduce_mean(loss2.item())
        cliploss_reduce = misc.all_reduce_mean(cliploss.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('test_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('test_loss1', loss1_reduce, epoch_1000x)
            log_writer.add_scalar('test_loss2', loss2_reduce, epoch_1000x)
            log_writer.add_scalar('test_cliploss', cliploss_reduce, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Test Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}