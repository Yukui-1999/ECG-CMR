import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import argparse
import os
import pandas as pd

def compare_correlations(y_true, y_pred1, y_pred2):
    # 计算皮尔逊相关系数
    r1, _ = pearsonr(y_true.flatten(), y_pred1.flatten())
    r2, _ = pearsonr(y_true.flatten(), y_pred2.flatten())
    
    # Fisher z转化
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # 计算标准误差
    n = len(y_true)
    se = np.sqrt(1 / (n - 3) * 2)

    # 计算z值和p值
    z_diff = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_diff)))

    return r1, r2, p_value

# 示例调用
# y_true = np.array([[...]])
# y_pred1 = np.array([[...]])
# y_pred2 = np.array([[...]])
# r1, r2, p_value = compare_correlations(y_true, y_pred1, y_pred2)


def get_args_parser():
    parser = argparse.ArgumentParser('Val_GenCMR', add_help=False)
    parser.add_argument('--downtask',default='UKB_reg',type=str,help="Path to the save csv file")
    parser.add_argument('--y_true',default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse_nopretrained/testUKB_metric/y_true.npy',type=str,help="Path to the test data .npy file")
    parser.add_argument('--y_pred1',default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse_nopretrained/testUKB_metric/y_pred.npy',type=str,help="Path to the test data .npy file") # Unaligned ECG   model1
    parser.add_argument('--y_pred2',default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both_headfcs/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeTrue/testUKB_metric/y_pred.npy',type=str,help="Path to the test data .npy file") # Aligned ECG with Frozen Encoder    model2
    parser.add_argument('--y_pred3',default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/vit_large_patchX/finetuneECG_both/ep400_40_lr1e-4_bs10_wd0.05_regression_EF1_freezeFalse/testUKB_metric/y_pred.npy',type=str,help="Path to the test data .npy file") # Fully Finetuned Aligned ECG    model3
    parser.add_argument('--y_pred4',default='/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/green_mim_swin_base_patch4_dec512b1/CMR_sup/cmr_modeboth_ep400_40_lr1e-4_bs32_wd0.05_regression_EF1/testUKB_metric/y_pred.npy',type=str,help="Path to the test data .npy file") # CMR Baseline     model4
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    y_true = np.load(args.y_true).mean(axis=1)
    y_pred1 = np.load(args.y_pred1).mean(axis=1)
    y_pred2 = np.load(args.y_pred2).mean(axis=1)
    y_pred3 = np.load(args.y_pred3).mean(axis=1)
    y_pred4 = np.load(args.y_pred4).mean(axis=1)

    print(f'y_true: {y_true.shape}')
    print(f'y_pred1: {y_pred1.shape}')
    print(f'y_pred2: {y_pred2.shape}')
    print(f'y_pred3: {y_pred3.shape}')
    print(f'y_pred4: {y_pred4.shape}')

    result = {
            'model1_vs_model2': 0,
            'model1_vs_model3': 0,
            'model1_vs_model4': 0,
            'model2_vs_model3': 0,
            'model2_vs_model4': 0,
            'model3_vs_model4': 0
        }
    r1, r2, p_value = compare_correlations(y_true, y_pred1, y_pred2)
    print(f'p_value: {p_value}')
    result['model1_vs_model2'] = p_value

    r1, r2, p_value = compare_correlations(y_true, y_pred1, y_pred3)
    print(f'p_value: {p_value}')
    result['model1_vs_model3'] = p_value

    r1, r2, p_value = compare_correlations(y_true, y_pred1, y_pred4)
    print(f'p_value: {p_value}')
    result['model1_vs_model4'] = p_value

    r1, r2, p_value = compare_correlations(y_true, y_pred2, y_pred3)
    print(f'p_value: {p_value}')
    result['model2_vs_model3'] = p_value

    r1, r2, p_value = compare_correlations(y_true, y_pred2, y_pred4)
    print(f'p_value: {p_value}')
    result['model2_vs_model4'] = p_value

    r1, r2, p_value = compare_correlations(y_true, y_pred3, y_pred4)
    print(f'p_value: {p_value}')
    result['model3_vs_model4'] = p_value

    dataframe = pd.DataFrame(result, index=[0])
    dataframe.to_csv(os.path.join('/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/final_result/compare_Pvalue',f'{args.downtask}_correlation.csv'), index=False)