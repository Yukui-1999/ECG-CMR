import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt

def cal_Pearson_correlation_with_bootstrap(y_real, y_fake, n_bootstraps=1000, ci=95, cor_index=None,agrs=None):
    save_path = os.path.dirname(args.y_fake)
    # 加载真实和生成的数据
    y_real = np.load(y_real)
    y_fake = np.load(y_fake)
    print(f'y_real:{y_real.shape}, y_fake:{y_fake.shape}')
    
    output = np.array(y_fake)
    label = np.array(y_real)
    
    # 存储每个指标的 Pearson 相关系数和 p 值
    corr_list = []
    p_value_list = []
    ci_lower_list = []
    ci_upper_list = []
    
    fig = plt.figure(figsize=(25, 17 * 5))
    # 对每个指标（共82个）进行 bootstrap 计算
    for i in range(82):
        corr, p_value = pearsonr(output[:, i], label[:, i])
        corr_list.append(corr)
        p_value_list.append(p_value)
        
        # 绘制散点图
        ax = fig.add_subplot(17, 5, i + 1)
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
        
        # 使用 bootstrap 方法来计算置信区间
        bootstrapped_corrs = []
        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrap for indicator {i+1}/82"):
            # 有放回地对样本进行采样
            indices = resample(range(len(output)), replace=True)
            corr_bootstrap, _ = pearsonr(output[indices, i], label[indices, i])
            bootstrapped_corrs.append(corr_bootstrap)
        
        # 计算置信区间
        lower = np.percentile(bootstrapped_corrs, (100 - ci) / 2)
        upper = np.percentile(bootstrapped_corrs, 100 - (100 - ci) / 2)
        ci_lower_list.append(lower)
        ci_upper_list.append(upper)
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(save_path, "regression.png"))
    
    return corr_list, p_value_list, ci_lower_list, ci_upper_list

def get_args_parser():
    parser = argparse.ArgumentParser('Val_GenCMR', add_help=False)
    parser.add_argument('--y_real', default=None, type=str, help="Path to the real data .npy file")
    parser.add_argument('--y_fake', default=None, type=str, help="Path to the fake data .npy file")
    parser.add_argument('--n_bootstraps', default=1000, type=int, help="Number of bootstrap samples")
    parser.add_argument('--ci', default=95, type=float, help="Confidence interval percentage")
    return parser

if __name__ == '__main__':
    cor_index = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume', 'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta maximum area', 'Ascending aorta minimum area', 'Ascending aorta distensibility', 'Descending aorta maximum area', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1', 'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6', 'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11', 'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16', 'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9', 'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2', 'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7', 'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12', 'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global', 'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5', 'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    
    args = get_args_parser()
    args = args.parse_args()

    # 计算 Pearson 相关系数及其置信区间
    corr_list, p_value_list, ci_lower_list, ci_upper_list = cal_Pearson_correlation_with_bootstrap(
        args.y_real, args.y_fake, n_bootstraps=args.n_bootstraps, ci=args.ci, cor_index=cor_index, args=args
    )
    
    # 保存结果
    save_path = os.path.dirname(args.y_fake)
    data = pd.DataFrame({
        'Pearson correlation coefficient': corr_list,
        'P-value': p_value_list,
        'CI lower': ci_lower_list,
        'CI upper': ci_upper_list
    }, index=cor_index)
    
    data.to_csv(os.path.join(save_path, "Relative_correlation_with_CI.csv"))
