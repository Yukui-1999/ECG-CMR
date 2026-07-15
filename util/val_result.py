import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import argparse
import scipy.stats as stats
import os
from util.Val_mutiClass import calculate_ovr_metrics
from util.Val_calss import calculate_metrics
import csv
from tqdm import tqdm

def save_multiclass_metrics_to_csv(metrics, filename, n_classes):
    # 动态生成列名
    columns = []
    for i in range(n_classes):
        columns.extend([
            f"Sensitivity_{i}", f"Specificity_{i}", f"Accuracy_{i}", f"AUC_{i}", 
            f"PPV_{i}", f"NPV_{i}", f"F1-Score_{i}"
        ])

    # 整理数据
    rows = []
    # 只有一行，因为每个类的指标值都在一行中
    row = []
    
    for i in range(n_classes):
        row.extend([
            f"{metrics[f'Class_{i}']['Sensitivity'][0]}, ({metrics[f'Class_{i}']['Sensitivity'][1][0]}, {metrics[f'Class_{i}']['Sensitivity'][1][1]})",
            f"{metrics[f'Class_{i}']['Specificity'][0]}, ({metrics[f'Class_{i}']['Specificity'][1][0]}, {metrics[f'Class_{i}']['Specificity'][1][1]})",
            f"{metrics[f'Class_{i}']['Accuracy'][0]}, ({metrics[f'Class_{i}']['Accuracy'][1][0]}, {metrics[f'Class_{i}']['Accuracy'][1][1]})",
            f"{metrics[f'Class_{i}']['AUC'][0]}, ({metrics[f'Class_{i}']['AUC'][1][0]}, {metrics[f'Class_{i}']['AUC'][1][1]})",
            f"{metrics[f'Class_{i}']['PPV'][0]}, ({metrics[f'Class_{i}']['PPV'][1][0]}, {metrics[f'Class_{i}']['PPV'][1][1]})",
            f"{metrics[f'Class_{i}']['NPV'][0]}, ({metrics[f'Class_{i}']['NPV'][1][0]}, {metrics[f'Class_{i}']['NPV'][1][1]})",
            f"{metrics[f'Class_{i}']['F1-Score'][0]}, ({metrics[f'Class_{i}']['F1-Score'][1][0]}, {metrics[f'Class_{i}']['F1-Score'][1][1]})"
        ])

    rows.append(row)  # 添加数据行

    # 写入CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)  # 写入列名
        writer.writerows(rows)    # 写入数据行

    print(f"Metrics saved to {filename}")


def save_metrics_to_csv(metrics, filename):
    # 需要的列名
    columns = ["Sensitivity", "Specificity", "Accuracy", "AUC", "PPV", "NPV", "F1-Score"]

    # 整理数据
    rows = []
    
    # 第一行: 平均值
    rows.append([
        f"{metrics['Sensitivity'][0]}{metrics['Sensitivity'][1]}",
        f"{metrics['Specificity'][0]}{metrics['Specificity'][1]}",
        f"{metrics['Accuracy'][0]}{metrics['Accuracy'][1]}",
        f"{metrics['AUC'][0]}({metrics['AUC'][1]}, {metrics['AUC'][2]})",  # AUC的特殊处理
        f"{metrics['PPV'][0]}{metrics['PPV'][1]}",
        f"{metrics['NPV'][0]}{metrics['NPV'][1]}",
        f"{metrics['F1-Score'][0]}({metrics['F1-Score'][1][0]}, {metrics['F1-Score'][1][1]})"
    ])

    # 写入CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)  # 写入列名
        writer.writerows(rows)    # 写入数据行

    print(f"Metrics saved to {filename}")


def process_val_result(y_true, y_pred, args):

    if args.downtask_type == 'BCE':
        # save y_true and y_pred
        np.savez_compressed(os.path.join(args.metric_save_path, "y_true_pred.npz"), y_true=y_true, y_pred=y_pred)
        metrics = calculate_metrics(y_true, y_pred)

        save_metrics_to_csv(metrics, os.path.join(args.metric_save_path, f"metrics.csv"))


    elif args.downtask_type == 'CE':
       
        # save y_true and y_pred
        np.savez_compressed(os.path.join(args.metric_save_path, "y_true_pred.npz"), y_true=y_true, y_pred=y_pred)
        metrics = calculate_ovr_metrics(y_true, y_pred)
        save_multiclass_metrics_to_csv(metrics, os.path.join(args.metric_save_path, f"metrics.csv"), y_true.shape[1])

    elif args.downtask_type == 'Regression':  
       
        n_bootstraps=1000
        Pearsonr, p = stats.pearsonr(y_true, y_pred)
        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrap for pearsonr"):
            # 有放回地对样本进行采样
            indices = resample(range(len(y_pred)), replace=True)
            corr_bootstrap, _ = stats.pearsonr(y_pred[indices], y_true[indices])
        
        # 计算置信区间
        ci_lower, ci_upper = np.percentile(corr_bootstrap, [2.5, 97.5])
        # save y_true and y_pred
        np.savez_compressed(os.path.join(args.metric_save_path, "y_true_pred.npz"), y_true=y_true, y_pred=y_pred)
        # Pearsonr, p
        # 'Pearsonr,(ci_lower, ci_upper)', 'p'
        metrics = {
            'Pearsonr': [(Pearsonr, (ci_lower, ci_upper))],
            'p': [p]
        }
        metrics = pd.DataFrame(metrics, index=[0])
        metrics.to_csv(os.path.join(args.metric_save_path, f"metrics.csv"))