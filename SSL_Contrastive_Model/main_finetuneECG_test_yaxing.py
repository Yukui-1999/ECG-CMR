import argparse
from typing import Tuple
import numpy as np
import os
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import modeling.ECGEncoder_co as ECGEncoder
import modeling.ECGEncoder as ECGEncoder_noco
from util.mutimodal_dataset_sl import get_train_dataset_class,get_test_dataset_class
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
import pickle
import seaborn as sns
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_custom_confusion_matrix_with_percent(output_class, label, class_names, save_path):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(label, output_class, labels=[0, 1, 2])
    
    # 计算行和列的总和
    row_sums = conf_matrix.sum(axis=1)  # 每一行的总数
    col_sums = conf_matrix.sum(axis=0)  # 每一列的总数
    diag_elements = np.diag(conf_matrix)  # 对角线元素（分类正确的数目）
    
    # 计算行和列的百分比
    row_percent = np.round(diag_elements / row_sums * 100, 1)  # 行百分比（右侧的数字）
    col_percent = np.round(diag_elements / col_sums * 100, 1)  # 列百分比（上方的数字）
    
    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)

    # 设置x轴和y轴标签
    row_num = [f'{class_names[i]} ({int(row_sum)})' for i, row_sum in enumerate(row_sums)]
    col_num = [f'{class_names[i]} ({int(col_sum)})' for i, col_sum in enumerate(col_sums)]
    ax.set_xticklabels(col_num, rotation=0)
    ax.set_yticklabels(row_num, rotation=0)
 

    # 在上方添加列百分比
    for i, col_sum in enumerate(col_sums):
        ax.text(i + 0.5, -0.1, f'{col_percent[i]}% ({diag_elements[i]}/{int(col_sum)})', 
                ha='center', va='center', fontsize=10, color='black')

    # 在右方添加行百分比
    for i, row_sum in enumerate(row_sums):
        ax.text(3.5, i + 0.5, f'{row_percent[i]}% ({diag_elements[i]}/{int(row_sum)})', 
                ha='center', va='center', fontsize=10, color='black')

    # 设置标题
    plt.title('Confusion matrix for classification of cardiomyopathy subtypes',pad=20)

    # 保存图片
   
    plt.xlabel('CardiacNets Prediction')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# def decision_curve_analysis(y_true, y_prob, thresholds,args):
#     net_benefits = []
#     epsilon = 1e-10  # 添加一个非常小的值来避免除以零
#     for threshold in thresholds:
#         tp = np.sum((y_prob >= threshold) & (y_true == 1))
#         fp = np.sum((y_prob >= threshold) & (y_true == 0))
#         fn = np.sum((y_prob < threshold) & (y_true == 1))
#         tn = np.sum((y_prob < threshold) & (y_true == 0))
        
#         net_benefit = tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold + epsilon))
#         net_benefits.append(net_benefit)

#     # 绘制决策曲线
#     plt.figure()
#     plt.plot(thresholds, net_benefits, color='blue', lw=2, label='Model')
#     plt.plot([0, 1], [0, 0], color='grey', lw=1, linestyle='--', label='No Model')
#     plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], color='red', lw=1, linestyle='--', label='All Positives')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([min(net_benefits) - 0.01, max(net_benefits) + 0.01])
#     plt.xlabel('Threshold Probability')
#     plt.ylabel('Net Benefit')
#     plt.title('Decision Curve Analysis')
#     plt.legend(loc="best")
    
#     plt.savefig(os.path.join(args.test_savepath, "decision_curve.svg"))
#     np.save(os.path.join(args.test_savepath, "net_benefit.npy"), np.array(net_benefit))


def decision_curve_analysis(label_binarized, output_prob, thresholds, args):
    num_classes = label_binarized.shape[1]
    plt.figure()

    for i in range(num_classes):
        net_benefits = []
        y_true = label_binarized[:, i]
        y_prob = output_prob[:, i]
        epsilon = 1e-10  # 添加一个非常小的值来避免除以零
        
        for threshold in thresholds:
            tp = np.sum((y_prob >= threshold) & (y_true == 1))
            fp = np.sum((y_prob >= threshold) & (y_true == 0))
            fn = np.sum((y_prob < threshold) & (y_true == 1))
            tn = np.sum((y_prob < threshold) & (y_true == 0))

            net_benefit = tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold + epsilon))
            net_benefits.append(net_benefit)

        # 绘制每个类别的决策曲线
        plt.plot(thresholds, net_benefits, lw=2, label=f'Class {i} Model')

    # 添加参考线：No Model 和 All Positives
    plt.plot([0, 1], [0, 0], color='grey', lw=1, linestyle='--', label='No Model')
    plt.plot([0, 1], [np.mean(label_binarized), np.mean(label_binarized)], color='red', lw=1, linestyle='--', label='All Positives')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([min(net_benefits) - 0.01, max(net_benefits) + 0.01])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc="best")
    
    # 保存决策曲线和净收益数据
    plt.savefig(os.path.join(args.test_savepath, "decision_curve.svg"))
    np.save(os.path.join(args.test_savepath, "net_benefit.npy"), np.array(net_benefits))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def roc_curve_plot(label_binarized, output_prob, args, auc_list):
    plt.figure()
    for i in range(label_binarized.shape[1]):
        fpr, tpr, _ = roc_curve(label_binarized[:, i], output_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_list[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')  # 画一条对角线（随机猜测的表现）
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.test_savepath, "roc_curves.svg"))
    plt.show()

import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve

def roc_curve_plot_(label_binarized, output_prob, args, auc_list):
    num_classes = label_binarized.shape[1]  # 获取类别数量
    plt.figure(figsize=(15, 5))  # 设置画布大小，每个子图大小可调
    
    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)  # 创建子图
        fpr, tpr, _ = roc_curve(label_binarized[:, i], output_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_list[i]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 画对角线（随机猜测的表现）
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i}')
        plt.legend(loc="lower right")
    
    # 保存整个画布（包含所有子图）
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(args.test_savepath, "roc_curves_per_class.svg"))
    plt.show()

# 示例调用
# roc_curve_plot(label_binarized, output_prob, args, auc_list)

# def roc_curve_plot(label, output_prob,args,auc_value):
#     # Calculate ROC curve
#     fpr, tpr, _ = roc_curve(label, output_prob)

#     # Plot ROC curve
#     plt.figure()
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
#     plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC)')
#     plt.legend(loc="lower right")
#     plt.savefig(os.path.join(args.test_savepath, "roc_curve.svg"))
#     np.save(os.path.join(args.test_savepath, "output_prob.npy"), output_prob)

def plot_calibration_curve(y_true, y_prob, args,i):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    # 计算Brier得分
    brier_score = brier_score_loss(y_true, y_prob)
    print(f"Brier score: {brier_score:.4f}")

    # 绘制校准曲线
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(f'Calibration Curve_{i}')
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.test_savepath, f'calibration_curve_{i}.svg'))
    np.save(os.path.join(args.test_savepath, f"CC_prob_true_{i}.npy"), prob_true)
    np.save(os.path.join(args.test_savepath, f"CC_prob_pred_{i}.npy"), prob_pred)
    return brier_score

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# python main_finetuneECG_test.py --downstream 'regression' --ecg_model 'vit_large_patchX' --device 'cuda:3' --classification_dis 'I21' --ecg_pretrained_model "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG/ep400_40_lr1e-4_bs10_wd0.05_regression_EF_freezeFalse/checkpoint-27-correlation-0.42.pth"
def get_args_parser():
    parser = argparse.ArgumentParser('ECG finetune test', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    
    # downstream task
    parser.add_argument('--downstream', default='yaxing', type=str, help='downstream task')
    parser.add_argument('--regression_dim', default=3, type=int, help='regression_dim')
    parser.add_argument('--classification_dis', default='I21', type=str, help='classification_dis')
    parser.add_argument('--condition_dim', default=3, type=int)
    # Model parameters
    parser.add_argument('--latent_dim', default=2048, type=int, metavar='N',
                        help='latent_dim')
    parser.add_argument('--ECGencoder_withco', default=True, type=str2bool,help='with_co or not')
    # ECG Model parameters
    parser.add_argument('--threshold', default=0.5, type=float,help='threshold')
    parser.add_argument('--ecg_model', default='vit_large_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',
                        default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG/ep400_40_lr1e-4_bs10_wd0.05_regression_EF_freezeFalse/checkpoint-27-correlation-0.42.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--ecg_input_channels', type=int, default=1, metavar='N',
                        help='ecginput_channels')
    parser.add_argument('--ecg_input_electrodes', type=int, default=12, metavar='N',
                        help='ecg input electrodes')
    parser.add_argument('--ecg_time_steps', type=int, default=5000, metavar='N',
                        help='ecg input length')
    parser.add_argument('--ecg_input_size', default=(12, 5000), type=Tuple,
                        help='ecg input size')
    parser.add_argument('--ecg_patch_height', type=int, default=1, metavar='N',
                        help='ecg patch height')
    parser.add_argument('--ecg_patch_width', type=int, default=100, metavar='N',
                        help='ecg patch width')
    parser.add_argument('--ecg_patch_size', default=(1, 100), type=Tuple,
                        help='ecg patch size')
    parser.add_argument('--ecg_globle_pool', default=False, type=str2bool, help='ecg_globle_pool')
    parser.add_argument('--ecg_drop_out', default=0.0, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # Augmentation parameters
    parser.add_argument('--input_size', type=tuple, default=(12, 5000))

    parser.add_argument('--timeFlip', type=float, default=0.33)

    parser.add_argument('--signFlip', type=float, default=0.33)
    
    # Dataset parameters
    parser.add_argument('--dataset',default='mutimodal_dataset_zheyi_ECG',type=str)
    parser.add_argument('--train_data_path',
                        default="/mnt/data2/ECG_CMR/trainval_onlyecg_data_dict_v11_dn.pt",
                        type=str,
                        help='dataset path')

    parser.add_argument('--test_data_path',
                        default="/mnt/data2/ECG_CMR/zheyi_data/Final_data/Fianl_ECGCMR_v2.pkl",
                        type=str,
                        help='test dataset path')
    
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--scaler', default='/mnt/data2/ECG_CMR/zheyi_data/Final_data/zheyi_scaler.pkl', type=str)
    return parser

@torch.no_grad()
def main(args):
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    device = torch.device(args.device)
    if args.downstream == 'classification':
        args.regression_dim = 1
    elif args.downstream == 'yaxing':
        args.regression_dim = 3

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # load data
    # if args.dataset == 'mutimodal_dataset_ECG':
    #     dataset_train = get_train_dataset_class(args.dataset,args)
    #     train_scaler = dataset_train.get_scaler()
    #     dataset_test = get_test_dataset_class(args.dataset,args,train_scaler)
    # elif args.dataset == 'mutimodal_dataset_zheyi_ECG':
    #     dataset_test = get_test_dataset_class(args.dataset,args,None)
    train_scaler = pickle.load(open(args.scaler, 'rb'))
    dataset_test = get_test_dataset_class(args.dataset,args,train_scaler)
    

    if args.ECGencoder_withco:
        model = ECGEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.regression_dim,
                drop_rate=args.ecg_drop_out,
                condition_dim=args.condition_dim,
                args=args,
            )
    elif args.ECGencoder_withco == False:
        model = ECGEncoder_noco.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.regression_dim,
                drop_rate=args.ecg_drop_out,
                args=args,
            )
    print("load pretrained ecg_model")
    ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
    ecg_checkpoint_model = ecg_checkpoint['model']
    msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
    print(msg)
    model.to(device)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )
    
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    output = []
    label = []
    loss = []

    for i, batch in enumerate(tqdm(data_loader_test, desc="Testing Progress")):
        
        ecg = batch['ecg'].unsqueeze(1).float().to(device, non_blocking=True)
        cond = batch['select_tar'].float().to(device, non_blocking=True)
        cha = batch['cha'].long().to(device)
        if torch.isnan(ecg).any():
            print("Warning: 11111 ecg contains NaN")
            # 找到 NaN 的位置
            nan_indices = torch.nonzero(torch.isnan(ecg), as_tuple=True)
            
            # 打印 NaN 的位置
            print("NaN indices:")
            for index in zip(*nan_indices):
                print(index)

            # 如果需要，输出具体的 NaN 值和它们的位置
            for index in zip(*nan_indices):
                print(f"NaN value at index {index}: {ecg[index]}")
            exit()

        with torch.cuda.amp.autocast():
            if torch.isnan(ecg).any():
                print("Warning: ecg contains NaN")
                exit()
            if torch.isnan(cond).any():
                print("Warning: cond contains NaN")
                exit()
            if torch.isnan(cha).any():
                print("Warning: cha contains NaN")
                exit()
            _, out = model(ecg,cond)
            loss_value = loss_fn(out, cha)
            # 检查 loss_value 是否包含 NaN
            if torch.isnan(loss_value).any():
                print("Warning: loss_value contains NaN")
                exit()
            else:
                pass

        # print(out.shape)
        loss.append(loss_value.item())
        label.append(cha.cpu().detach().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)

    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    print(output.shape)
    print(label.shape)
    print(f"test loss:{np.mean(loss)}")
    output_prob = torch.nn.functional.softmax(torch.tensor(output).float(), dim=-1).numpy()

    num_classes = len(np.unique(label))
    metrics = {}

    # Binarize the labels for OvR approach
    label_binarized = label_binarize(label, classes=np.arange(num_classes))

    np.save(os.path.join(args.test_savepath, "y_pred.npy"), output_prob)
    np.save(os.path.join(args.test_savepath, "y_true.npy"), label_binarized)
    
    # Calculate AUC for each class
    auc_list = []
    for i in range(num_classes):
        auc = roc_auc_score(label_binarized[:, i], output_prob[:, i])
        print(f"Class {i} AUC: {auc}")
        metrics[f'AUC_Class_{i}'] = auc
        auc_list.append(auc)

    # Plot ROC curve for each class (you would modify roc_curve_plot to handle multi-class)
    roc_curve_plot_(label_binarized, output_prob, args, auc_list)

    # Calculate decision curve analysis
    thresholds = np.linspace(0, 1, 100)
    decision_curve_analysis(label_binarized, output_prob, thresholds, args)

    # Calculate calibration curve (similar to the binary case, but for each class)
    brier_scores = []
    for i in range(num_classes):
        brier_score = plot_calibration_curve(label_binarized[:, i], output_prob[:, i], args, i)
        print(f"Class {i} Brier Score: {brier_score}")
        metrics[f'Brier_Score_Class_{i}'] = brier_score
        brier_scores.append(brier_score)

    # Convert probabilities to predicted class labels
    output_class = np.argmax(output_prob, axis=1)

    # Calculate overall accuracy
    acc = accuracy_score(label, output_class)
    plot_custom_confusion_matrix_with_percent(output_class, label, ['RCM', 'DCM', 'HCM'],os.path.join(args.test_savepath, "confusion_matrix.svg"))
    print(f"Overall Accuracy: {acc}")
    metrics['Accuracy'] = acc

    # Calculate Precision, Recall, F1 Score, and Specificity for each class
    for i in range(num_classes):
        precision = precision_score(label, output_class, labels=[i], average='macro')
        recall = recall_score(label, output_class, labels=[i], average='macro')
        f1 = f1_score(label, output_class, labels=[i], average='macro')
        
        tn, fp, fn, tp = confusion_matrix(label_binarized[:, i], output_class == i).ravel()
        specificity = tn / (tn + fp)
        
        print(f"Class {i} Precision: {precision}")
        print(f"Class {i} Recall: {recall}")
        print(f"Class {i} F1 Score: {f1}")
        print(f"Class {i} Specificity: {specificity}")
        
        metrics[f'Precision_Class_{i}'] = precision
        metrics[f'Recall_Class_{i}'] = recall
        metrics[f'F1_Score_Class_{i}'] = f1
        metrics[f'Specificity_Class_{i}'] = specificity

    # Save all metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.test_savepath, "classification_metrics.csv"), index=False)

    
    print('test done')


    return 0

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.suffix:
        args.test_savepath = os.path.join(os.path.dirname(args.ecg_pretrained_model),"test"+args.suffix)
    else:
        args.test_savepath = os.path.join(os.path.dirname(args.ecg_pretrained_model),"test")
    if args.test_savepath:
        Path(args.test_savepath).mkdir(parents=True, exist_ok=True)
    main(args)
