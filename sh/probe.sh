#!/bin/bash

# defining and training the Probe model and save as 'probe_model.pth'
python probe_model.py

# prompting to microsoft/Phi-3-mini-4k-instruct and obtain the results
python probe_prompt.py
# results will be stored at csv file in ./probe_results folder after run.

# computing ECE, AUROC, AUPRC-Positive, AUPRC-Negative
python probe_metrics.py

# visualization of result; confidence-count and confidence-accuracy with Probe
python probe_visual.py