import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import argparse
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt

def cal_Pearson_correlation_singlePic(y_real, y_fake):
    y_real = np.load(y_real)
    y_fake = np.load(y_fake)
    n_bootstraps = 1000
    ci = 95
    print(f'y_real:{y_real.shape}, y_fake:{y_fake.shape}')
    output = np.array(y_fake)
    label = np.array(y_real)
    
    corr_list = []
    p_value_list = []
    ci_lower_list = []
    ci_upper_list = []
    
    for i in tqdm(range(82)):
        # 每个子图单独创建
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算相关系数和p值
        corr, p_value = pearsonr(output[:, i].flatten(), label[:, i].flatten())
        corr_list.append(corr)
        p_value_list.append(p_value)

        label_flat = label[:, i].flatten()
        output_flat = output[:, i].flatten()
        
        # 绘制散点图
        ax.scatter(label_flat, output_flat, s=10, c='blue', alpha=0.5)

        label_flat = label_flat.astype(np.float32)
        output_flat = output_flat.astype(np.float32)
        
        # 拟合线性回归线
        fit = np.polyfit(label_flat, output_flat, 1)
        x = np.linspace(min(label_flat), max(label_flat), 100)
        y = np.polyval(fit, x)
        
        # 绘制拟合线
        ax.plot(x, y, color='black')
        
        # 在图上添加相关系数的文本
        ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes,fontsize=16)
        ax.text(0.1, 0.8, f'P-value: {p_value:.2e}', transform=ax.transAxes,fontsize=16)
        # 设置子图的标题
        ax.set_title(cor_index[i],fontsize=24)
        ax.set_xlabel('Ground Truth',fontsize=20)
        ax.set_ylabel('Predict',fontsize=20)

        # Bootstrap 计算置信区间
        # bootstrapped_corrs = []
        # for _ in tqdm(range(n_bootstraps), desc=f"Bootstrap for indicator {i+1}/82"):
        #     indices = resample(range(len(output)), replace=True)
        #     corr_bootstrap, _ = pearsonr(output[indices, i], label[indices, i])
        #     bootstrapped_corrs.append(corr_bootstrap)

        # lower = np.percentile(bootstrapped_corrs, (100 - ci) / 2)
        # upper = np.percentile(bootstrapped_corrs, 100 - (100 - ci) / 2)
        # ci_lower_list.append(lower)
        # ci_upper_list.append(upper)

        # 保存每个子图
        single_pic_dir = os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/Project_version2/GreenMIM/SupFig/Fig1','single_pic_update')
        if not os.path.exists(single_pic_dir):
            os.makedirs(single_pic_dir)
        save_path = os.path.join(single_pic_dir, f"{cor_index[i]}.svg")
        plt.savefig(save_path)
        
        # 关闭当前图，避免内存占用过多
        plt.close(fig)
    
    return corr_list, p_value_list, ci_lower_list, ci_upper_list

def cal_Pearson_correlation(y_real, y_fake):
    y_real = np.load(y_real)
    y_fake = np.load(y_fake)
    n_bootstraps = 1000
    ci = 95
    print(f'y_rea:{y_real.shape}, y_fake:{y_fake.shape}')
    output = np.array(y_fake)
    label = np.array(y_real)
    fig = plt.figure(figsize=(25, 17 * 5))
    corr_list = []
    p_value_list = []
    ci_lower_list = []
    ci_upper_list = []
    for i in tqdm(range(82)):
        ax = fig.add_subplot(17, 5, i + 1)
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
        ax.text(0.1, 0.9, f'Correlation: {corr:.2f}', transform=ax.transAxes,fontsize=16)
        ax.text(0.1, 0.8, f'P-value: {p_value:.2e}', transform=ax.transAxes,fontsize=16)
        # 设置子图的标题
        ax.set_title(cor_index[i],fontsize=24)
        ax.set_xlabel('Ground Truth',fontsize=20)
        ax.set_ylabel('Predict',fontsize=20)

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

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join('/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/final_result/Gen_CMR_val', "regression_4.png"))
    return corr_list, p_value_list,ci_lower_list,ci_upper_list

def get_args_parser():
    parser = argparse.ArgumentParser('Val_GenCMR', add_help=False)
    parser.add_argument('--y_real',default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/green_mim_swin_base_patch4_dec512b1/CMR_sup_GenEval/cmr_modeboth_ep400_40_lr1e-4_bs32_wd0.05_regression_EF1_IsRealFalse/test/y_pred.npy", type=str)#
    parser.add_argument('--y_fake',default="/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/green_mim_swin_base_patch4_dec512b1/CMR_sup_GenEval/cmr_modeboth_ep400_40_lr1e-4_bs32_wd0.05_regression_EF1_withCondFalse_IsRealFalse/test/y_pred.npy", type=str)#
    return parser
if __name__ == '__main__':
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    args = get_args_parser()
    args = args.parse_args()
    corr_list, p_value_list,ci_lower_list,ci_upper_list = cal_Pearson_correlation_singlePic(args.y_real, args.y_fake)
    exit()
    args.savepath = os.path.dirname(args.y_fake)
    data = pd.DataFrame(data={'Pearson correlation coefficient': corr_list, 
                              'P-value': p_value_list,
                              'ci_lower_list':ci_lower_list,
                              'ci_upper_list':ci_upper_list}, index=cor_index)
    data.to_csv(os.path.join('/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/final_result/Gen_CMR_val',"Relative_correlation_4.csv"))