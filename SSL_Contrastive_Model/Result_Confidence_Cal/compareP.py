import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
import argparse
import os
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd

def sensitivity_comparison_two_models(preds_A, preds_B, actual, threshold=0.5):
    # Helper function to calculate TP and FN
    def calculate_tp_fn(y_pred, y_true, threshold):
        y_pred_bin = (y_pred >= threshold).astype(int)  # Convert probability to binary labels
        tp = np.sum((y_pred_bin == 1) & (y_true == 1))  # True Positives
        fn = np.sum((y_pred_bin == 0) & (y_true == 1))  # False Negatives
        return np.array([tp, fn])
    
    # Calculate TP and FN for both models
    tp_fn_model_A = calculate_tp_fn(preds_A, actual, threshold)
    tp_fn_model_B = calculate_tp_fn(preds_B, actual, threshold)
    
    # Calculate sensitivity for each model
    sensitivities = {}
    sensitivities['Model_A'] = tp_fn_model_A[0] / tp_fn_model_A.sum() if tp_fn_model_A.sum() > 0 else 0
    sensitivities['Model_B'] = tp_fn_model_B[0] / tp_fn_model_B.sum() if tp_fn_model_B.sum() > 0 else 0
    
    # Group TP and FN data into successes and total observations for z-test
    successes = np.array([tp_fn_model_A[0], tp_fn_model_B[0]])
    nobs = np.array([tp_fn_model_A.sum(), tp_fn_model_B.sum()])
    
    # Perform z-test between the two models
    z_stat, p_value = proportions_ztest(count=successes, nobs=nobs)
    
    # Output the sensitivities, z-statistic, and p-value
    return z_stat, p_value


def specificity_comparison_two_models(preds_A, preds_B, actual, threshold=0.5):
    # Helper function to calculate TN and FP
    def calculate_tn_fp(y_pred, y_true, threshold):
        y_pred_bin = (y_pred >= threshold).astype(int)  # Convert probability to binary labels
        tn = np.sum((y_pred_bin == 0) & (y_true == 0))  # True Negatives
        fp = np.sum((y_pred_bin == 1) & (y_true == 0))  # False Positives
        return np.array([tn, fp])
    
    # Calculate TN and FP for both models
    tn_fp_model_A = calculate_tn_fp(preds_A, actual, threshold)
    tn_fp_model_B = calculate_tn_fp(preds_B, actual, threshold)
    
    # Calculate specificity for each model
    specificities = {}
    specificities['Model_A'] = tn_fp_model_A[0] / tn_fp_model_A.sum() if tn_fp_model_A.sum() > 0 else 0
    specificities['Model_B'] = tn_fp_model_B[0] / tn_fp_model_B.sum() if tn_fp_model_B.sum() > 0 else 0
    
    # Group TN and FP data into successes (True Negatives) and total observations for z-test
    successes = np.array([tn_fp_model_A[0], tn_fp_model_B[0]])
    nobs = np.array([tn_fp_model_A.sum(), tn_fp_model_B.sum()])
    
    # Perform z-test between the two models
    z_stat, p_value = proportions_ztest(count=successes, nobs=nobs)
    
    # Output the specificities, z-statistic, and p-value
    return z_stat, p_value

def accuracy_comparison_two_models(preds_A, preds_B, actual, threshold=0.5):
    # Helper function to calculate TP, TN, FP, and FN
    def calculate_tp_tn_fp_fn(y_pred, y_true, threshold):
        y_pred_bin = (y_pred >= threshold).astype(int)  # Convert probability to binary labels
        tp = np.sum((y_pred_bin == 1) & (y_true == 1))  # True Positives
        tn = np.sum((y_pred_bin == 0) & (y_true == 0))  # True Negatives
        fp = np.sum((y_pred_bin == 1) & (y_true == 0))  # False Positives
        fn = np.sum((y_pred_bin == 0) & (y_true == 1))  # False Negatives
        return tp, tn, fp, fn
    
    # Calculate TP, TN, FP, and FN for both models
    tp_model_A, tn_model_A, fp_model_A, fn_model_A = calculate_tp_tn_fp_fn(preds_A, actual, threshold)
    tp_model_B, tn_model_B, fp_model_B, fn_model_B = calculate_tp_tn_fp_fn(preds_B, actual, threshold)
    
    # Calculate accuracy for each model
    accuracies = {}
    total_A = tp_model_A + tn_model_A + fp_model_A + fn_model_A
    total_B = tp_model_B + tn_model_B + fp_model_B + fn_model_B
    accuracies['Model_A'] = (tp_model_A + tn_model_A) / total_A if total_A > 0 else 0
    accuracies['Model_B'] = (tp_model_B + tn_model_B) / total_B if total_B > 0 else 0
    
    # Group correct predictions (TP + TN) and total samples for z-test
    successes = np.array([tp_model_A + tn_model_A, tp_model_B + tn_model_B])
    nobs = np.array([total_A, total_B])
    
    # Perform z-test between the two models
    z_stat, p_value = proportions_ztest(count=successes, nobs=nobs)
    
    # Output the accuracies, z-statistic, and p-value
    return z_stat, p_value


def f1_comparison_two_models(preds_A, preds_B, actual, threshold=0.5):
    # Helper function to calculate TP, FP, and FN
    def calculate_tp_fp_fn(y_pred, y_true, threshold):
        y_pred_bin = (y_pred >= threshold).astype(int)  # Convert probability to binary labels
        tp = np.sum((y_pred_bin == 1) & (y_true == 1))  # True Positives
        fp = np.sum((y_pred_bin == 1) & (y_true == 0))  # False Positives
        fn = np.sum((y_pred_bin == 0) & (y_true == 1))  # False Negatives
        return tp, fp, fn
    
    # Calculate TP, FP, and FN for both models
    tp_model_A, fp_model_A, fn_model_A = calculate_tp_fp_fn(preds_A, actual, threshold)
    tp_model_B, fp_model_B, fn_model_B = calculate_tp_fp_fn(preds_B, actual, threshold)
    
    # Calculate precision and recall for each model
    precision_A = tp_model_A / (tp_model_A + fp_model_A) if (tp_model_A + fp_model_A) > 0 else 0
    recall_A = tp_model_A / (tp_model_A + fn_model_A) if (tp_model_A + fn_model_A) > 0 else 0
    
    precision_B = tp_model_B / (tp_model_B + fp_model_B) if (tp_model_B + fp_model_B) > 0 else 0
    recall_B = tp_model_B / (tp_model_B + fn_model_B) if (tp_model_B + fn_model_B) > 0 else 0
    
    # Calculate F1 score for each model
    f1_scores = {}
    f1_scores['Model_A'] = 2 * (precision_A * recall_A) / (precision_A + recall_A) if (precision_A + recall_A) > 0 else 0
    f1_scores['Model_B'] = 2 * (precision_B * recall_B) / (precision_B + recall_B) if (precision_B + recall_B) > 0 else 0
    
    # For z-test, we will consider the F1 score as a proportion test based on TP, FP, and FN
    # However, z-test might not be the most appropriate for F1 comparison as it is a combination of precision and recall.
    # Perform z-test between TP of the two models
    successes = np.array([tp_model_A, tp_model_B])
    nobs = np.array([tp_model_A + fp_model_A + fn_model_A, tp_model_B + fp_model_B + fn_model_B])
    
    z_stat, p_value = proportions_ztest(count=successes, nobs=nobs)
    
    # Output the F1 scores, z-statistic, and p-value
    return z_stat, p_value


class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        # self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold :print("There is a significant difference")
        else:        print("There is NO significant difference")

    def return_result(self):
        z,p=self._compute_z_p()
        return z,p

# Model A (random) vs. "good" model B
# preds_A = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# preds_B = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
# actual=    np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
# DelongTest(preds_A,preds_B,actual)


def get_args_parser():
    parser = argparse.ArgumentParser('Val_GenCMR', add_help=False)
    parser.add_argument('--downtask',default='UKB_CAD',type=str,help="Path to the save csv file")
    parser.add_argument('--y_true',default=None,type=str,help="Path to the test data .npy file")
    parser.add_argument('--y_pred1',default=None,type=str,help="Path to the test data .npy file") # Unaligned ECG   model1
    parser.add_argument('--y_pred2',default=None,type=str,help="Path to the test data .npy file") # Aligned ECG with Frozen Encoder    model2
    parser.add_argument('--y_pred3',default=None,type=str,help="Path to the test data .npy file") # Fully Finetuned Aligned ECG    model3
    parser.add_argument('--y_pred4',default=None,type=str,help="Path to the test data .npy file") # CMR Baseline     model4
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    y_true = np.load(args.y_true)
    y_pred1 = np.load(args.y_pred1)
    y_pred2 = np.load(args.y_pred2)
    y_pred3 = np.load(args.y_pred3)
    if args.y_pred4 is not None:
        y_pred4 = np.load(args.y_pred4)
    print(f'y_true: {y_true.shape}')
    print(f'y_pred1: {y_pred1.shape}')
    print(f'y_pred2: {y_pred2.shape}')
    print(f'y_pred3: {y_pred3.shape}')
    if args.y_pred4 is not None:
        print(f'y_pred4: {y_pred4.shape}')
    # exit()
    if args.y_pred4 is not None:
        result = {
            'model1_vs_model2':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model1_vs_model3':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model1_vs_model4':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model2_vs_model3':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model2_vs_model4':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model3_vs_model4':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            }
        }
        # Model 1 vs. Model 2
        z_stat, p_value = DelongTest(y_pred1, y_pred2, y_true).return_result()
        result['model1_vs_model2']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['f1'].append(p_value)

        # Model 1 vs. Model 3
        z_stat, p_value = DelongTest(y_pred1, y_pred3, y_true).return_result()
        result['model1_vs_model3']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['f1'].append(p_value)

        # Model 1 vs. Model 4
        z_stat, p_value = DelongTest(y_pred1, y_pred4, y_true).return_result()
        result['model1_vs_model4']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred1, y_pred4, y_true)
        result['model1_vs_model4']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred1, y_pred4, y_true)
        result['model1_vs_model4']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred1, y_pred4, y_true)
        result['model1_vs_model4']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred1, y_pred4, y_true)
        result['model1_vs_model4']['f1'].append(p_value)

        # Model 2 vs. Model 3
        z_stat, p_value = DelongTest(y_pred2, y_pred3, y_true).return_result()
        result['model2_vs_model3']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['f1'].append(p_value)

        # Model 2 vs. Model 4
        z_stat, p_value = DelongTest(y_pred2, y_pred4, y_true).return_result()
        result['model2_vs_model4']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred2, y_pred4, y_true)
        result['model2_vs_model4']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred2, y_pred4, y_true)
        result['model2_vs_model4']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred2, y_pred4, y_true)
        result['model2_vs_model4']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred2, y_pred4, y_true)
        result['model2_vs_model4']['f1'].append(p_value)

        # Model 3 vs. Model 4
        z_stat, p_value = DelongTest(y_pred3, y_pred4, y_true).return_result()
        result['model3_vs_model4']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred3, y_pred4, y_true)
        result['model3_vs_model4']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred3, y_pred4, y_true)
        result['model3_vs_model4']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred3, y_pred4, y_true)
        result['model3_vs_model4']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred3, y_pred4, y_true)
        result['model3_vs_model4']['f1'].append(p_value)



    elif args.y_pred4 is None:
        result = {
            'model1_vs_model2':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model1_vs_model3':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            },
            'model2_vs_model3':{
                'auc':[],
                'sensitivity':[],
                'Specificity':[],
                'accuracy':[],
                'f1':[]
            }
        }
        # Model 1 vs. Model 2
        z_stat, p_value = DelongTest(y_pred1, y_pred2, y_true).return_result()
        result['model1_vs_model2']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred1, y_pred2, y_true)
        result['model1_vs_model2']['f1'].append(p_value)

        # Model 1 vs. Model 3
        z_stat, p_value = DelongTest(y_pred1, y_pred3, y_true).return_result()
        result['model1_vs_model3']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred1, y_pred3, y_true)
        result['model1_vs_model3']['f1'].append(p_value)


        # Model 2 vs. Model 3
        z_stat, p_value = DelongTest(y_pred2, y_pred3, y_true).return_result()
        result['model2_vs_model3']['auc'].append(p_value)
        z_stat, p_value = sensitivity_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['sensitivity'].append(p_value)
        z_stat, p_value = specificity_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['Specificity'].append(p_value)
        z_stat, p_value = accuracy_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['accuracy'].append(p_value)
        z_stat, p_value = f1_comparison_two_models(y_pred2, y_pred3, y_true)
        result['model2_vs_model3']['f1'].append(p_value)

    


    result_df = pd.DataFrame.from_dict({(i,j): result[i][j] 
                            for i in result.keys() 
                            for j in result[i].keys()},
                        orient='index')

    # Save to CSV
    result_df.to_csv(os.path.join('/mnt/data2/dingzhengyao/work/checkpoint/Newproject_v1/ckpts/final_result/compare_Pvalue', f'{args.downtask}_compare_Pvalue.csv'))










































