#!/bin/bash

python 'tf_evaluation.py' --model-name "Llama-2-7b" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "Llama-2-7b-chat" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "Llama-2-13b" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "Llama-2-13b-chat" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "Llama-2-70b" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "Llama-2-70b-chat" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "gpt-3.5-turbo" --data-path "tf_scope_evaluation.jsonl"
python 'tf_evaluation.py' --model-name "gpt-4" --data-path "tf_scope_evaluation.jsonl"