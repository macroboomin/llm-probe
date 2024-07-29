import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_metrics(df):
    # Expected Calibration Error (ECE)
    def compute_ece(df, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = df[(df['probe_normalized'] > bin_lower) & (df['probe_normalized'] <= bin_upper)]
            prop_in_bin = len(in_bin) / len(df)
            if len(in_bin) > 0:
                avg_confidence_in_bin = in_bin['probe_normalized'].mean()
                avg_accuracy_in_bin = in_bin['correct'].mean()
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
        
        return ece

    # AUROC
    def compute_auroc(df):
        return roc_auc_score(df['correct'], df['probe'])

    # AUPRC-Positive (PR-P)
    def compute_pr_p(df):
        precision, recall, _ = precision_recall_curve(df['correct'], df['probe'])
        return auc(recall, precision)
    
    # AUPRC-Negative (PR-N)
    def compute_pr_n(df):
        precision, recall, _ = precision_recall_curve(1 - df['correct'], df['probe'])
        return auc(recall, precision)

    # Accuracy
    def compute_accuracy(df):
        return df['correct'].mean()

    # Normalize confidence for ECE calculation
    df['probe_normalized'] = df['probe'] / 100

    ece = compute_ece(df) * 100
    auroc = compute_auroc(df) * 100
    pr_p = compute_pr_p(df) * 100
    pr_n = compute_pr_n(df) * 100
    accuracy = compute_accuracy(df) * 100

    return round(ece, 1), round(auroc, 1), round(pr_p, 1), round(pr_n, 1), round(accuracy, 1)

# Load the CSV files
col_math = pd.read_csv('./probe_results/col_math.csv')
biz_ethics = pd.read_csv('./probe_results/biz_ethics.csv')
prf_law = pd.read_csv('./probe_results/prf_law.csv')

# Compute metrics for each dataset
metrics_col_math = compute_metrics(col_math)
metrics_biz_ethics = compute_metrics(biz_ethics)
metrics_prf_law = compute_metrics(prf_law)

# Calculate the number of rows in each dataset
num_rows_col_math = len(col_math)
num_rows_biz_ethics = len(biz_ethics)
num_rows_prf_law = len(prf_law)

# Calculate total number of rows
total_rows = num_rows_col_math + num_rows_biz_ethics + num_rows_prf_law

# Calculate weighted average metrics
avg_ece = round((metrics_col_math[0] * num_rows_col_math + metrics_biz_ethics[0] * num_rows_biz_ethics + metrics_prf_law[0] * num_rows_prf_law) / total_rows, 1)
avg_auroc = round((metrics_col_math[1] * num_rows_col_math + metrics_biz_ethics[1] * num_rows_biz_ethics + metrics_prf_law[1] * num_rows_prf_law) / total_rows, 1)
avg_pr_p = round((metrics_col_math[2] * num_rows_col_math + metrics_biz_ethics[2] * num_rows_biz_ethics + metrics_prf_law[2] * num_rows_prf_law) / total_rows, 1)
avg_pr_n = round((metrics_col_math[3] * num_rows_col_math + metrics_biz_ethics[3] * num_rows_biz_ethics + metrics_prf_law[3] * num_rows_prf_law) / total_rows, 1)
avg_accuracy = round((metrics_col_math[4] * num_rows_col_math + metrics_biz_ethics[4] * num_rows_biz_ethics + metrics_prf_law[4] * num_rows_prf_law) / total_rows, 1)

results = pd.DataFrame({
    'Metric': ['ECE', 'AUROC', 'PR-P', 'PR-N', 'Accuracy'],
    'College Mathematics': metrics_col_math,
    'Business Ethics': metrics_biz_ethics,
    'Professional Law': metrics_prf_law,
    'Average': [avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy]
})

results.to_csv('./probe_results/probe_metrics.csv', index=False)

# Output metrics
print("College Mathematics Probe Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_col_math)
print("Business Ethics Probe Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_biz_ethics)
print("Professional Law Probe Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_prf_law)

print("\nAverage Probe Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy)
