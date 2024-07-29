#!/bin/bash

# prompt strategy = vanilla_prompt (verbalized_confidence)
# no sampling and no aggregation

# prompting to microsoft/Phi-3-mini-4k-instruct and obtain the results
python vanlilla_prompt.py
# results will be stored at csv file in ./verbalized_results folder after run.

# computing ECE, AUROC, AUPRC-Positive, AUPRC-Negative
python verbalized_metrics.py

# visualization of result; confidence-count and confidence-accuracy
python verbalized_visual.py