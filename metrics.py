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
            in_bin = df[(df['confidence_normalized'] > bin_lower) & (df['confidence_normalized'] <= bin_upper)]
            prop_in_bin = len(in_bin) / len(df)
            if len(in_bin) > 0:
                avg_confidence_in_bin = in_bin['confidence_normalized'].mean()
                avg_accuracy_in_bin = in_bin['correct'].mean()
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
        
        return ece

    # AUROC
    def compute_auroc(df):
        return roc_auc_score(df['correct'], df['confidence'])

    # AUPRC-Positive (PR-P)
    def compute_pr_p(df):
        precision, recall, _ = precision_recall_curve(df['correct'], df['confidence'])
        return auc(recall, precision)
    
    # AUPRC-Negative (PR-N)
    def compute_pr_n(df):
        precision, recall, _ = precision_recall_curve(1 - df['correct'], df['confidence'])
        return auc(recall, precision)

    # Normalize confidence for ECE calculation
    df['confidence_normalized'] = df['confidence'] / 100

    ece = compute_ece(df) * 100
    auroc = compute_auroc(df) * 100
    pr_p = compute_pr_p(df) * 100
    pr_n = compute_pr_n(df) * 100

    return round(ece, 1), round(auroc, 1), round(pr_p, 1), round(pr_n, 1)

# Load the CSV files
biz_ethics = pd.read_csv('./data/Biz_Ethics_processed.csv')
gsm8k = pd.read_csv('./data/GSM8K_processed.csv')
prf_law = pd.read_csv('./data/Prf_Law_processed.csv')

# Compute metrics for each dataset
metrics_gsm8k = compute_metrics(gsm8k)
metrics_biz_ethics = compute_metrics(biz_ethics)
metrics_prf_law = compute_metrics(prf_law)

# Calculate average metrics
avg_ece = round(np.mean([metrics_gsm8k[0], metrics_biz_ethics[0], metrics_prf_law[0]]), 1)
avg_auroc = round(np.mean([metrics_gsm8k[1], metrics_biz_ethics[1], metrics_prf_law[1]]), 1)
avg_pr_p = round(np.mean([metrics_gsm8k[2], metrics_biz_ethics[2], metrics_prf_law[2]]), 1)
avg_pr_n = round(np.mean([metrics_gsm8k[3], metrics_biz_ethics[3], metrics_prf_law[3]]), 1)

# Output metrics
print("GSM8K Metrics: ECE, AUROC, PR-P, PR-N")
print(metrics_gsm8k)
print("Business Ethics Metrics: ECE, AUROC, PR-P, PR-N")
print(metrics_biz_ethics)
print("Professional Law Metrics: ECE, AUROC, PR-P, PR-N")
print(metrics_prf_law)

print("\nAverage Metrics: ECE, AUROC, PR-P, PR-N")
print(avg_ece, avg_auroc, avg_pr_p, avg_pr_n)
