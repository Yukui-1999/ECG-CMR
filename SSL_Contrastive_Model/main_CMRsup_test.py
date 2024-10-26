import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path
import sys
import torch
import torch.backends.cudnn as cudnn
import timm
import torch.nn as nn
from sklearn.utils import resample
from util.mutimodal_dataset_sl import get_train_dataset_class,get_test_dataset_class
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve
from modeling.Config import Config_swin_base,Config_swin_base_win14,Config_swin_large_win14
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from modeling.swin_transformer_other import build_swin
from modeling.swin_transformer_both import SwinTransformerBoth
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
# from engine_pretrain import train_one_epoch, evaluate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decision_curve_analysis(y_true, y_prob, thresholds,args):
    net_benefits = []
    epsilon = 1e-10  # 添加一个非常小的值来避免除以零
    for threshold in thresholds:
        tp = np.sum((y_prob >= threshold) & (y_true == 1))
        fp = np.sum((y_prob >= threshold) & (y_true == 0))
        fn = np.sum((y_prob < threshold) & (y_true == 1))
        tn = np.sum((y_prob < threshold) & (y_true == 0))
        
        net_benefit = tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold + epsilon))
        net_benefits.append(net_benefit)

    # 绘制决策曲线
    plt.figure()
    plt.plot(thresholds, net_benefits, color='blue', lw=2, label='Model')
    plt.plot([0, 1], [0, 0], color='grey', lw=1, linestyle='--', label='No Model')
    plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], color='red', lw=1, linestyle='--', label='All Positives')
    plt.xlim([0.0, 1.0])
    plt.ylim([min(net_benefits) - 0.01, max(net_benefits) + 0.01])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc="best")
    
    plt.savefig(os.path.join(args.test_savepath, "decision_curve.png"))
    np.save(os.path.join(args.test_savepath, "net_benefit.npy"), np.array(net_benefit))


def roc_curve_plot(label, output_prob,args,auc_value):
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(label, output_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.test_savepath, "roc_curve.png"))
    np.save(os.path.join(args.test_savepath, "output_prob.npy"), output_prob)

def plot_calibration_curve(y_true, y_prob, args):
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
    plt.title('Calibration Curve')
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.test_savepath, "calibration_curve.png"))
    np.save(os.path.join(args.test_savepath, "CC_prob_true.npy"), prob_true)
    np.save(os.path.join(args.test_savepath, "CC_prob_pred.npy"), prob_pred)
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

# python main_CMRsup_test.py --train_data_path "/data1/dingzhengyao/ECG_CMR/data/trainval_onlycmr_data_dict_v11_dn.pt" --test_data_path "/data1/dingzhengyao/ECG_CMR/data/test_onlycmr_data_dict_v11.pt" --dataset 'mutimodal_dataset_CMR' --cmr_mode 'cmr' --downstream 'regression' --classification_dis  'I21' --cmr_pretrained_model "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/green_mim_swin_base_patch4_dec512b1/CMR_sup/cmr_modela_cmr_ep400_40_lr1e-4_bs64_wd0.05_regression_EF1/checkpoint-38-correlation-0.58.pth"
# python main_CMRsup_test.py --dataset 'mutimodal_dataset_CMR' --cmr_mode 'cmr' --downstream 'classification' --classification_dis  'I21' --cmr_pretrained_model "/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/cmr_tiaozhiI21c1/checkpoint-14-auc-0.79.pth"
# python main_CMRsup_test.py --dataset 'mutimodal_dataset_CMR' --cmr_mode 'cmr' --downstream 'classification' --classification_dis  'I42' --cmr_pretrained_model "/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/cmr_tiaozhiI42c1/checkpoint-3-auc-0.63.pth"
# python main_CMRsup_test.py --dataset 'mutimodal_dataset_CMR' --cmr_mode 'cmr' --downstream 'classification' --classification_dis  'I48' --cmr_pretrained_model "/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/cmr_tiaozhiI48c1/checkpoint-16-auc-0.78.pth"
# python main_CMRsup_test.py --dataset 'mutimodal_dataset_CMR' --cmr_mode 'cmr' --downstream 'classification' --classification_dis  'I50' --cmr_pretrained_model "/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/cmr_tiaozhiI50c1/checkpoint-15-auc-0.78.pth"


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    #downstream task
    parser.add_argument('--resizeshape',default=224,type=int,help='resize shape')
    parser.add_argument('--downstream', default='regression', type=str, help='downstream task')
    parser.add_argument('--regression_dim',default=82,type=int,help='regression_dim')
    parser.add_argument('--classification_dis', default='I21', type=str, help='classification_dis')
    # Model parameters
    parser.add_argument('--latent_dim', default=256, type=int, metavar='N',
                        help='latent_dim')
    
    parser.add_argument('--cmr_mode',default='both',type=str,help='CMR_mode')
    parser.add_argument('--model', default='green_mim_swin_base_patch4_dec512b1', type=str, metavar='MODEL',
                        help='Name of model to train')

    # CMR Model parameters
    parser.add_argument('--with_cond',default=False,type=str2bool,help='with_cond')
    parser.add_argument('--cmr_pretrained_model',
                        default="/mnt/data/dingzhengyao/work/checkpoint/preject_version1/cmr_pretrain_output_dir/cmr_tiaozhif1/checkpoint-116-correlation-0.55.pth",
                        type=str)
    parser.add_argument('--img_size', default=80, type=int, metavar='N', help='img_size of cmr')
    

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=10, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0.05, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters

    parser.add_argument('--train_data_path',
                        default="/mnt/data2/ECG_CMR/trainval_data_dict_v11.pt",
                        type=str,
                        help='dataset path')
    parser.add_argument('--test_data_path',
                        default="/mnt/data2/ECG_CMR/test_data_dict_v11.pt",
                        type=str,
                        help='test dataset path')
    parser.add_argument('--dataset',default='mutimodal_dataset_laCMR',type=str)
    parser.add_argument('--cmr_isreal',default=True,type=str2bool,help='cmr_isreal')

    parser.add_argument('--output_dir', default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/",
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True, 
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--scaler', default=None, type=str)
    return parser

@torch.no_grad()
def main(args):
    
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    

    
    cmr_model_config = Config_swin_base()
    if args.downstream == 'classification':
        cmr_model_config.MODEL.NUM_CLASSES = 1
    
    
    if args.cmr_mode == 'cmr' or args.cmr_mode == 'la_cmr':
        model = build_swin(cmr_model_config)
        model.to(device)
    else:
        args.pretrained_swin = None
        model = SwinTransformerBoth(config=cmr_model_config,args=args)
        model.to(device)

   
    print("load pretrained cmr_model")
    checkpoint = torch.load(args.cmr_pretrained_model,map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'])
    print(msg)
    model.to(args.device)

    # if args.dataset == 'mutimodal_dataset_Gen_CMRlaCMR':
    #     dataset_train = get_train_dataset_class(args.dataset,args)
    #     scaler = dataset_train.get_scaler()
    #     save_path = '/mnt/data2/dingzhengyao/work/checkpoint/ECG_CMR/diffusion/experiments/scaler.pkl'
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     pickle.dump(scaler, open(save_path, 'wb'))
    #     dataset_test = get_test_dataset_class(args.dataset,args,scaler)
    # else:
    print('load dataset')
    if args.scaler is not None:
        train_scaler = pickle.load(open(args.scaler, 'rb'))
        dataset_test = get_test_dataset_class(args.dataset,args,train_scaler)
    else:
        dataset_test = get_test_dataset_class(args.dataset,args,None)

    
    print(f'test set len:{len(dataset_test)}')

    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    model.eval()
    loss_fn = torch.nn.MSELoss()
    loss = []
    output = []
    label = []
    for i, batch in enumerate(tqdm(data_loader_test, desc="Testing Progress")):

        if args.cmr_mode == 'both':
            cmr = batch['cmr'].float().to(device, non_blocking=True)
            la_cmr = batch['la_cmr'].float().to(device, non_blocking=True)
        else:
            cmr = batch[args.cmr_mode].float().to(device, non_blocking=True)
        cond = batch['select_tar'].float().to(device, non_blocking=True)
        cha = batch['cha'].float().to(device)
        
        if args.dataset == 'mutimodal_dataset_NEWECG':
            cha = cha.unsqueeze(1)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif args.downstream == 'classification':
            cha = batch[args.classification_dis]
            cha = np.array([1 if x != '0' else 0 for x in cha])
            cha = torch.from_numpy(cha).float().to(device)
            cha = cha.unsqueeze(1)
            loss_fn = torch.nn.BCEWithLogitsLoss()


        with torch.cuda.amp.autocast():
            if args.cmr_mode == 'both':
                out = model(cmr, la_cmr, cond)
            else:
                out = model(cmr, cond)
            loss_value = loss_fn(out, cha)
            

        loss.append(loss_value.item())
        label.append(cha.cpu().detach().numpy())
        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        output.append(out)
    print(output)
    output = np.concatenate(output, axis=0)
    label = np.concatenate(label, axis=0)
    print(output.shape)
    print(label.shape)


    print(f"test loss:{np.mean(loss)}")
    if args.downstream == 'classification':
        metrics = {}
        output_prob = sigmoid(output)
        np.save(os.path.join(args.test_savepath, "y_pred.npy"), output_prob)
        np.save(os.path.join(args.test_savepath, "y_true.npy"), label)
        # Calculate ROC AUC
        auc = roc_auc_score(label, output_prob)
        print(f"test auc: {auc}")
        metrics['AUC'] = auc
        roc_curve_plot(label, output_prob,args,auc)

        # Calculate decision curve analysis
        thresholds = np.linspace(0, 1, 100)
        decision_curve_analysis(label, output_prob, thresholds, args)

        # Calculate calibration curve
        brier_score = plot_calibration_curve(label, output_prob, args)
        metrics['Brier Score'] = brier_score

        # Binarize the output with a threshold of 0.5
        output_binary = (output_prob > 0.5).astype(int)

        # Calculate accuracy
        acc = accuracy_score(label, output_binary)
        print(f"test accuracy: {acc}")
        metrics['Accuracy'] = acc

        # Calculate precision
        precision = precision_score(label, output_binary)
        print(f"test precision: {precision}")
        metrics['Precision'] = precision

        # Calculate recall
        recall = recall_score(label, output_binary)
        print(f"test recall: {recall}")
        metrics['Recall'] = recall

        # Calculate F1 score
        f1 = f1_score(label, output_binary)
        print(f"test F1 score: {f1}")
        metrics['F1 Score'] = f1

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(label, output_binary).ravel()

        # Calculate specificity
        specificity = tn / (tn + fp)
        print(f"test specificity: {specificity}")
        metrics['Specificity'] = specificity

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.test_savepath, "classification_metrics.csv"), index=False)
    if args.downstream == 'regression':

        output = train_scaler.inverse_transform(output)
        label = train_scaler.inverse_transform(label)
        n_bootstraps = 1000
        ci = 95
        np.save(os.path.join(args.test_savepath, "y_pred.npy"), output)
        np.save(os.path.join(args.test_savepath, "y_true.npy"), label)
        import matplotlib.pyplot as plt

        # 创建一个画布
        fig = plt.figure(figsize=(25, 17 * 5))
        corr_list = []
        p_value_list = []
        ci_lower_list = []
        ci_upper_list = []
        for i in range(82):
            # 创建一个子图
            ax = fig.add_subplot(17, 5, i + 1)
            # 计算相关系数
            # corr = np.corrcoef(output[:, i].flatten(), label[:, i].flatten())[0, 1]
            # 加上P值计算 待更改
            corr, p_value = pearsonr(output[:, i].flatten(), label[:, i].flatten())
            corr_list.append(corr)
            p_value_list.append(p_value)

            label_flat = label[:, i].flatten()
            output_flat = output[:, i].flatten()
            # 绘制散点图
            ax.scatter(label_flat, output_flat)

            label_flat = label_flat.astype(np.float32)
            output_flat = output_flat.astype(np.float32)
            # Fit a line to the data
            fit = np.polyfit(label_flat, output_flat, 1)

            # Create a sequence of x values spanning the range of the data
            x = np.linspace(min(label_flat), max(label_flat), 100)

            # Use the polynomial fit to calculate the corresponding y values
            y = np.polyval(fit, x)

            # Plot the fit line
            ax.plot(x, y, color='black')

            # 在图上添加相关系数的文本
            ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes)
            ax.text(0.1, 0.8, f'P-value: {p_value:.2e}', transform=ax.transAxes)
            # 设置子图的标题
            ax.set_title(cor_index[i])
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predict')

            bootstrapped_corrs = []
            for _ in tqdm(range(n_bootstraps), desc=f"Bootstrap for indicator {i+1}/82"):
                # 有放回地对样本进行采样
                indices = resample(range(len(output)), replace=True)
                corr_bootstrap, _ = pearsonr(output[indices, i], label[indices, i])
                bootstrapped_corrs.append(corr_bootstrap)

            lower = np.percentile(bootstrapped_corrs, (100 - ci) / 2)
            upper = np.percentile(bootstrapped_corrs, 100 - (100 - ci) / 2)
            ci_lower_list.append(lower)
            ci_upper_list.append(upper)

        # 显示图形
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(args.test_savepath,"regression.png"))
        # data2 = pd.DataFrame(data=corr_list, index=cor_index, columns=['Pearson correlation coefficient'])
        data2 = pd.DataFrame(data={'Pearson correlation coefficient': corr_list, 
                                   'P-value': p_value_list,
                                   'CI lower': ci_lower_list,
                                    'CI upper': ci_upper_list}, index=cor_index)
        # PATH为导出文件的路径和文件名
        data2.to_csv(os.path.join(args.test_savepath,"regression.csv"))
    print('test done')


    return 0

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.suffix:
        args.test_savepath = os.path.join(os.path.dirname(args.cmr_pretrained_model),"test"+args.suffix)
    else:
        args.test_savepath = os.path.join(os.path.dirname(args.cmr_pretrained_model),"test")
    if args.test_savepath:
        Path(args.test_savepath).mkdir(parents=True, exist_ok=True)
    main(args)
