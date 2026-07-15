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
import os 
import torch
import torchmetrics
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
import util.misc as misc
import util.lr_sched as lr_sched
import pandas as pd

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
        
        targets = targets.to(device, non_blocking=True).long()

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


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             epoch: int = 0,
             use_amp: bool = True,
             num_classes: int = 3,  # specify the number of classes
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'valid:'
    
    # Initialize empty lists to store targets and outputs
    tgt_list = np.array([])  # True labels
    opt_list = np.array([])  # Predicted probabilities (probabilities per class)
    
    for sample, target in metric_logger.log_every(data_loader, 10, header):
        
        if args.use_gen_cmr:
            ecg = sample['ecg'].to(device, non_blocking=True).float()
            gen_cmr = sample['gen_cmr'].to(device, non_blocking=True).float()
        else:
            sample = sample.to(device, non_blocking=True).float()
        
        target = target.to(device, non_blocking=True).long()  # Assuming target is a multi-class label (not one-hot)

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
                if sample.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
                else:
                    output = model(sample)
            loss = criterion(output, target)
        
        # Concatenate predictions and targets
        if len(opt_list) == 0:
            opt_list = torch.softmax(output, dim=1).cpu().numpy()  # Apply softmax for multi-class probability
            tgt_list = target.cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.softmax(output, dim=1).cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.cpu().numpy()])
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    # Multi-class classification metrics calculation using OvR (One-vs-Rest)
    # Binarize the target labels for each class
    pred_labels = (opt_list.argmax(axis=1))  # Choose the class with the highest probability
    true_labels = tgt_list.squeeze().astype(int)

    # Initialize variables to store metrics for each class
    accuracies = []
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    f1_scores = []
    aucs = []
    
    # Calculate metrics for each class using OvR
    for class_idx in range(num_classes):
        binary_true = (true_labels == class_idx).astype(int)  # True labels for current class
        binary_pred = (pred_labels == class_idx).astype(int)  # Predicted labels for current class
        
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = safe_divide(tp, tp + fp, default=0.0)
        npv = safe_divide(tn, tn + fn, default=0.0)
        f1 = 2 * tp / (2 * tp + fp + fn)
        
        # AUC (One-vs-Rest)
        auc = roc_auc_score((true_labels == class_idx).astype(int), opt_list[:, class_idx])

        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
        ppvs.append(ppv)
        npvs.append(npv)
        f1_scores.append(f1)
        aucs.append(auc)
    
    
    
    metrics = {}
    class_name = ['RCM', 'DCM', 'HCM']
    for i in range(num_classes):
        metrics[f'Accuracy_{class_name[i]}'] = accuracies[i]
        metrics[f'Sensitivity_{class_name[i]}'] = sensitivities[i]
        metrics[f'Specificity_{class_name[i]}'] = specificities[i]
        metrics[f'PPV_{class_name[i]}'] = ppvs[i]
        metrics[f'NPV_{class_name[i]}'] = npvs[i]
        metrics[f'F1-Score_{class_name[i]}'] = f1_scores[i]
        metrics[f'AUC_{class_name[i]}'] = aucs[i]
    
    # Save metrics to CSV
    metrics = pd.DataFrame(metrics, index=[0])
    
    
    
    # Average metrics across all classes
    avg_acc = np.mean(accuracies)
    avg_sens = np.mean(sensitivities)
    avg_spec = np.mean(specificities)
    avg_ppv = np.mean(ppvs)
    avg_npv = np.mean(npvs)
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(aucs)
    
    metric_logger.synchronize_between_processes()
    
    # Log results
    print('* loss@all {losses.global_avg:.3f}, Acc {avg_acc:.3f}, Sens {avg_sens:.3f}, Spec {avg_spec:.3f}, PPV {avg_ppv:.3f}, NPV {avg_npv:.3f}, F1 {avg_f1:.3f}, AUC {avg_auc:.3f}'.format(
        losses=metric_logger.loss,
        avg_acc=avg_acc,
        avg_sens=avg_sens,
        avg_spec=avg_spec,
        avg_ppv=avg_ppv,
        avg_npv=avg_npv,
        avg_f1=avg_f1,
        avg_auc=avg_auc
    ))

    # Prepare test state and log dictionary
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['Acc'] = avg_acc
    test_state['Sens'] = avg_sens
    test_state['Spec'] = avg_spec
    test_state['PPV'] = avg_ppv
    test_state['NPV'] = avg_npv
    test_state['F1'] = avg_f1
    test_state['AUC'] = avg_auc
    
    if log_writer is not None:
        epoch_1000x = int((epoch) * 1000)
        log_writer.add_scalar('valid_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('Acc', avg_acc, epoch_1000x)
        log_writer.add_scalar('Sens', avg_sens, epoch_1000x)
        log_writer.add_scalar('Spec', avg_spec, epoch_1000x)
        log_writer.add_scalar('PPV', avg_ppv, epoch_1000x)
        log_writer.add_scalar('NPV', avg_npv, epoch_1000x)
        log_writer.add_scalar('F1', avg_f1, epoch_1000x)
        log_writer.add_scalar('AUC', avg_auc, epoch_1000x)

    
    return test_state, metrics, opt_list, tgt_list


