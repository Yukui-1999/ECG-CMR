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
import pickle
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_output_to_df(jilu_output, cor_index, jilu_label, jilu_binglihao):
    # 确保 jilu['output'] 的形状和 cor_index 长度一致
    if jilu_output.shape[1] != len(cor_index):
        raise ValueError("jilu['output'] 的列数与 cor_index 长度不匹配")

    # 创建 DataFrame，列名是 cor_index，数据是 jilu['output']
    df = pd.DataFrame(jilu_output, columns=cor_index)
    
    # 将 label 和 binglihao 添加为 DataFrame 的新列
    df['label'] = jilu_label
    df['binglihao'] = jilu_binglihao

    return df

def convert_softmaxoutput_to_df(jilu_output, cor_index, jilu_label, jilu_binglihao):
    # 将三分类概率分布数据分为三列
    output_tensor = torch.tensor(jilu_output).float()
    jilu_output_softmax = torch.nn.functional.softmax(output_tensor, dim=-1).numpy()
    
    # 创建 DataFrame
    df = pd.DataFrame(jilu_output_softmax, columns=['class_0_prob', 'class_1_prob', 'class_2_prob'])
    
    # 将 label 和 binglihao 添加为 DataFrame 的新列
    df['label'] = jilu_label
    df['binglihao'] = jilu_binglihao

    return df


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

# python main_finetuneECG_predict.py --downstream 'regression' --ecg_pretrained_model "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse/checkpoint-29-correlation-0.43.pth" --test_data_path '/mnt/data2/ECG_CMR/zheyi_data/Final_data/renjishiyan.pkl' --scaler "/mnt/data2/ECG_CMR/Cha_scaler_v11_dn.pkl" --suffix 'zheyi_renji_reg' --condition_dim 24
# python main_finetuneECG_predict.py --downstream 'classification'  --ecg_pretrained_model "/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both_Final_MIMIC_CM_aligned/ep400_40_lr1e-4_bs10_wd0.05_classification__freezeFalse_Sametarprocess_ECGencoder_withcoTrue/checkpoint-7-auc-0.86.pth" --test_data_path '/mnt/data2/ECG_CMR/zheyi_data/Final_data/renjishiyan.pkl' --scaler "/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/Sametarprocess/CM_scaler.pkl" --suffix 'zheyi_renji_bin' --condition_dim 3
# /mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/Sametarprocess/CM_scaler.pkl
# /mnt/data2/ECG_CMR/Cha_scaler_v11_dn.pkl
# /mnt/data2/ECG_CMR/zheyi_data/Final_data/zheyi_scaler.pkl
# /mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both_Final_MIMIC_CM_aligned/ep400_40_lr1e-4_bs10_wd0.05_classification__freezeFalse_Sametarprocess_ECGencoder_withcoTrue/checkpoint-7-auc-0.86.pth

def get_args_parser():
    parser = argparse.ArgumentParser('ECG finetune test', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    
    # downstream task
    parser.add_argument('--downstream', default='classification', type=str, help='downstream task')
    parser.add_argument('--regression_dim', default=82, type=int, help='regression_dim')
    parser.add_argument('--classification_dis', default='I42', type=str, help='classification_dis')
    parser.add_argument('--resizeshape',default=256,type=int,help='resize shape')
    # Model parameters
    parser.add_argument('--latent_dim', default=2048, type=int, metavar='N',
                        help='latent_dim')
   
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
    parser.add_argument('--ECGencoder_withco', default=True, type=str2bool,help='with_co or not')
    # Augmentation parameters
    parser.add_argument('--input_size', type=tuple, default=(12, 5000))

    parser.add_argument('--timeFlip', type=float, default=0.33)

    parser.add_argument('--signFlip', type=float, default=0.33)
    parser.add_argument('--condition_dim', default=3, type=int)
    # Dataset parameters
    parser.add_argument('--dataset',default='mutimodal_dataset_zheyi_ECG',type=str)
    parser.add_argument('--train_data_path',
                        default="/mnt/data2/ECG_CMR/trainval_onlyecg_data_dict_v11_dn.pt",
                        type=str,
                        help='dataset path')
    parser.add_argument('--training_percentage',default=1.0,type=float)
    parser.add_argument('--test_data_path',
                        default="/mnt/data2/ECG_CMR/test_onlyecg_data_dict_v11.pt",
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
    parser.add_argument('--scaler', default=None, type=str)

    return parser

@torch.no_grad()
def main(args):
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    device = torch.device(args.device)
    if args.downstream == 'classification':
        args.regression_dim = 1
    elif args.downstream == 'yaxing':
        args.regression_dim = 3
    elif args.downstream == 'regression':
        args.regression_dim = 82

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

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
    

    jilu = {
        'binglihao':[],
        'output' : [],
        'label': []
    }
    for i, batch in enumerate(tqdm(data_loader_test, desc="Testing Progress")):
        
        ecg = batch['ecg'].unsqueeze(1).float().to(device, non_blocking=True)
        cond = batch['select_tar'].float().to(device, non_blocking=True)
        cha = batch['cha'].float()
        binglihao = batch['binglihao']
        jilu['binglihao'].extend(binglihao)
        if args.downstream == 'classification' or args.downstream == 'yaxing':
            jilu['label'].extend(cha.numpy())
        with torch.cuda.amp.autocast():
            _, out = model(ecg,cond)

        out = out.cpu().detach().numpy()
        out = out.reshape(-1, out.shape[-1])  # reshape the output
        jilu['output'].extend(out)
    
    jilu['output'] = np.array(jilu['output'])
    print(jilu['output'].shape)


    if args.downstream == 'classification':
    
        jilu['output'] = np.squeeze(sigmoid(jilu['output']))
        metrics_df = pd.DataFrame(jilu)
        metrics_df.to_csv(os.path.join(args.test_savepath, "renji_result_bin.csv"), index=False)

    elif args.downstream == 'yaxing':
        jilu['output']  = torch.nn.functional.softmax(torch.tensor(jilu['output']).float(), dim=-1).numpy()
        metrics_df = convert_softmaxoutput_to_df(jilu['output'], None, jilu['label'], jilu['binglihao'])
        metrics_df.to_csv(os.path.join(args.test_savepath, "renji_result_three.csv"), index=False)

    elif args.downstream == 'regression':
        jilu['output'] = train_scaler.inverse_transform(jilu['output'])
        jilu['label'] = ['none'] * len(jilu['output'])
        metrics_df = convert_output_to_df(jilu['output'], cor_index, jilu['label'], jilu['binglihao'])
        metrics_df.to_csv(os.path.join(args.test_savepath, "renji_result_reg.csv"), index=False)
        # plt.show()
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
