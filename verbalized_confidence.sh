#!/bin/bash

# prompt strategy = vanilla_prompt (verbalized_confidence)
# no sampling and no aggregation

# prompting to microsoft/Phi-3-mini-4k-instruct and obtain the results
python vanlilla_prompt.py
# results will be stored at csv file in ./data folder after run.

# computing ECE, AUROC, AUPRC-Positive, AUPRC-Negative
python metrics.py

# visualization of result; confidence-count and confidence-accuracy
python visual.py