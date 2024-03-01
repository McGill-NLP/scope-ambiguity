#!/bin/bash

python 'liu_tf_evaluation.py' --model-name "Llama-2-7b" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "Llama-2-7b-chat" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "Llama-2-13b" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "Llama-2-13b-chat" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "Llama-2-70b" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "Llama-2-70b-chat" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "gpt-3.5-turbo" --data-path "liu_scope_evaluation.jsonl"
python 'liu_tf_evaluation.py' --model-name "gpt-4" --data-path "liu_scope_evaluation.jsonl"