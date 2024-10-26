import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import argparse
import scipy.stats as stats
import os

def bootstrap_auc(y_true, y_pred, n_bootstrap=10000, ci=0.95):
    """
    计算 AUC 及其 95% 置信区间（通过 bootstrap 方法）。
    
    参数:
    y_true -- 真实标签 (0 或 1)
    y_pred -- 预测概率
    n_bootstrap -- bootstrap 次数，默认10000
    ci -- 置信水平，默认 95%

    返回:
    auc -- 原始数据的 AUC
    ci_lower -- 置信区间下限
    ci_upper -- 置信区间上限
    """
    # 计算原始 AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # 存储 bootstrap AUC 结果
    bootstrapped_aucs = []
    
    # 随机数种子
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstrap):
        # 随机抽样（有放回）
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # 如果抽样中只有一个类别，则跳过该次计算
            continue
        
        bootstrapped_auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(bootstrapped_auc)
    
    # 计算置信区间
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()
    
    # 95% 置信区间
    ci_lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    ci_upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    
    return auc, ci_lower, ci_upper

def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    计算 TP, FP, TN, FN。
    
    参数:
    y_true -- 真实标签 (0 或 1)
    y_pred -- 预测概率
    threshold -- 用于将概率转换为标签的阈值，默认是 0.5
    
    返回:
    TP -- 真阳性
    FP -- 假阳性
    TN -- 真阴性
    FN -- 假阴性
    """
    # 根据阈值将预测的概率转为二分类标签
    y_pred_label = (y_pred >= threshold).astype(int)
    
    # 计算 TP, FP, TN, FN
    TP = np.sum((y_true == 1) & (y_pred_label == 1))
    FP = np.sum((y_true == 0) & (y_pred_label == 1))
    TN = np.sum((y_true == 0) & (y_pred_label == 0))
    FN = np.sum((y_true == 1) & (y_pred_label == 0))
    
    return TP, FP, TN, FN


def calculate_f1(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def simulate_f1(TP, FP, TN, FN, n_simulations=10000):
    f1_scores = []

    for _ in range(n_simulations):
        sim_TP = np.random.binomial(TP + FN, TP / (TP + FN))
        sim_FN = TP + FN - sim_TP
        sim_FP = np.random.binomial(TP + FP, FP / (TP + FP))
        sim_f1 = calculate_f1(sim_TP, sim_FP, sim_FN)
        f1_scores.append(sim_f1)

    return np.percentile(f1_scores, [2.5, 97.5])


def calculate_metrics(y_true, y_pred):
    # 计算基本指标
    TP, FP, TN, FN = calculate_confusion_matrix(y_true, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    total = TP + FP + TN + FN
    pe = ((TP + FN) * (TP + FP) + (FN + TN) * (FP + TN)) / (total ** 2)
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) != 0 else 0

    # 计算置信区间
    def wilson_score(p, n):
        z = stats.norm.ppf(0.975)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
        return (center - margin, center + margin)

    sensitivity_ci = wilson_score(sensitivity, TP + FN)
    specificity_ci = wilson_score(specificity, TN + FP)
    accuracy_ci = wilson_score(accuracy, total)
    PPV_ci = wilson_score(PPV, TP + FP) if (TP + FP) != 0 else (0, 0)
    NPV_ci = wilson_score(NPV, TN + FN) if (TN + FN) != 0 else (0, 0)
    F1_ci = simulate_f1(TP, FP, TN, FN)
    return {
        "Sensitivity":(sensitivity, sensitivity_ci),
        "Specificity":(specificity, specificity_ci),
        "Accuracy":(accuracy, accuracy_ci),
        "AUC":bootstrap_auc(y_true, y_pred),
        "PPV":(PPV, PPV_ci),
        "NPV":(NPV, NPV_ci),
        "F1-Score":(F1,F1_ci),
        "Kappa":kappa
    }



def get_args_parser():
    parser = argparse.ArgumentParser('Val_GenCMR', add_help=False)
    parser.add_argument('--test_path',default=None,type=str,help="Path to the test data .npy file")
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.y_true = os.path.join(args.test_path, "y_true.npy")
    args.y_pred = os.path.join(args.test_path, "y_pred.npy")
    y_true = np.load(args.y_true)
    y_pred = np.load(args.y_pred)
   
    metrics = calculate_metrics(y_true, y_pred)
    with open(os.path.join(args.test_path, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    tongyilujing = args.test_path.replace("/", "_") + "_metrics.txt"
    with open(os.path.join('/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/final_result/', tongyilujing), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    for key, value in metrics.items():
        print(f"{key}: {value}")