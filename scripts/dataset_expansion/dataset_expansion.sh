#!/bin/bash

python 'exp1_dataset_expansion.py' --openai-api-key "<OPENAI_API_KEY>" --source-dataset-filepath "<filepath to Experiment 1A dataset -- see README about access>" --generated-data-directory "<GENERATED_DATA_DIRECTORY>"
python 'exp2_dataset_expansion.py' --openai-api-key "<OPENAI_API_KEY>" --source-dataset-filepath "datasets/exp2a_base_dataset.csv" --generated-data-directory "<GENERATED_DATA_DIRECTORY>"
