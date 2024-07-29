import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from CSV files
probe_metrics = pd.read_csv('./probe_results/probe_metrics.csv')
verbalized_metrics = pd.read_csv('./verbalized_results/verbalized_metrics.csv')

# Extract the relevant metrics
metrics_to_compare = ['ECE', 'AUROC']

probe_metrics_filtered = probe_metrics[probe_metrics['Metric'].isin(metrics_to_compare)]
verbalized_metrics_filtered = verbalized_metrics[verbalized_metrics['Metric'].isin(metrics_to_compare)]

# Ensure the 'graph' directory exists
if not os.path.exists('graph'):
    os.makedirs('graph')

# Plot the comparison graphs as bar plots and save them
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

for ax, metric in zip(axes, metrics_to_compare):
    probe_data = probe_metrics_filtered.set_index('Metric').loc[metric]
    verbalized_data = verbalized_metrics_filtered.set_index('Metric').loc[metric]
    
    categories = probe_data.index.tolist()
    bar_width = 0.35
    index = range(len(categories))
    
    ax.bar(index, verbalized_data.values, bar_width, label='Verbalized', alpha=0.7)
    ax.bar([i + bar_width for i in index], probe_data.values, bar_width, label='Probe', alpha=0.7)
    
    ax.set_title(f'Comparison of {metric}')
    ax.set_ylabel(metric)
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# Save the figure to the 'graph' directory
fig.savefig('graph/comparison_metrics_bar.png')

plt.show()
