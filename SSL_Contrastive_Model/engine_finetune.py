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
        
        cha = batch['cha'].float().to(device)
        classification_dict = {'I21': batch['I21'], 'I42': batch['I42'], 'I48': batch['I48'], 'I50': batch['I50'], 'I08':batch['I08'], 'I25':batch['I25'], 'I34':batch['I34'], 'I35':batch['I35']}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        select_modal = batch[args.select_modal]
        select_modal = select_modal.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(select_modal)
            loss = loss_fn(output, cha)
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

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


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

       
        
        cha = batch['cha'].float().to(device)
        classification_dict = {'I21': batch['I21'], 'I42': batch['I42'], 'I48': batch['I48'], 'I50': batch['I50'], 'I08':batch['I08'], 'I25':batch['I25'], 'I34':batch['I34'], 'I35':batch['I35']}
        if args.downstream == 'classification':
            cha = classification_dict[args.classification_dis].float().to(device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        select_modal = batch[args.select_modal]
        select_modal = select_modal.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(select_modal)
            loss = loss_fn(out, cha)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        label.append(cha.cpu().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)

    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    if args.downstream == 'classification':
        auc = roc_auc_score(label, output)
        metric_logger.update(auc=auc)

    if args.downstream == 'regression':
        corr_list = []
        for i in range(82):
            corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            corr_list.append(corr)
        metric_logger.update(correlation=np.mean(corr_list))

    
    metric_logger.synchronize_between_processes()
    print("validation stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')
    


    loss_value_reduce = misc.all_reduce_mean(loss_value)
    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        if args.downstream == 'classification':
            log_writer.add_scalar('auc', auc, epoch_1000x)
        if args.downstream == 'regression':
            log_writer.add_scalar('correlation', np.mean(corr_list), epoch_1000x)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}