#!/bin/bash

python experiment2_filtering.py --path-to-human-results "../../human_results/exp2a_human_results_cleaned.csv" --output-path "../../datasets/exp2a_base_dataset.csv"
python experiment2_filtering.py --path-to-human-results "../../human_results/exp2b_human_results_cleaned.csv" --output-path "../../datasets/exp2b_base_dataset.csv"
