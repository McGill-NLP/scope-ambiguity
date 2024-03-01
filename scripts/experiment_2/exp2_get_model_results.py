import argparse
import os
import json
import gc 
import torch
import numpy as np 
import pandas as pd 
import openai
from transformers import BitsAndBytesConfig 
from modded_scorer import modded_scorer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai-api-key', type=str)
    parser.add_argument('--auth-token', type=str)
    parser.add_argument('--cache-dir', type=str)
    parser.add_argument('--path-to-model-list', type=str)
    parser.add_argument('--path-to-model-stimuli', type=str)
    parser.add_argument('--output-file-path', type=str)
    args = parser.parse_args()
    get_model_results(args)

def get_model_results(args):
    path_to_model_stimuli = args.path_to_model_stimuli
    path_to_model_list = args.path_to_model_list
    output_file_path = args.output_file_path
    openai_api_key = args.openai_api_key
    auth_token = args.auth_token # for gated models like Llama 2
    cache_dir = args.cache_dir # if specific cache directory required for disk space limit reasons
    
    with open(path_to_model_list, 'r') as file:
        data = file.read()
        model_list = json.loads(data)
    
    print("Models and details:\n", model_list)
    dataset_df = pd.read_csv(path_to_model_stimuli)
    sentences = dataset_df['sentence'].tolist()
    followups = dataset_df['followup'].tolist()
    print("Data ready!")
    for key in model_list.keys():
        print(key)
        model_info = model_list[key]
        model_name = model_info['model_name']
        quantization_bool = bool(model_info['quantization'])
        if model_info['source']=='huggingface':
            modelscorer = modded_scorer.IncrementalLMScorer(model_name, auth_token=auth_token, device='cuda', load_in_8bit=quantization_bool, cache_dir=cache_dir)
            responses = [sum(x) for x in modelscorer.compute_stats(modelscorer.prime_text(sentences, followups))]
            dataset_df[key] = responses
            del(modelscorer.model)
            gc.collect()
            torch.cuda.empty_cache()
        
        '''
        Deprecated! Since OpenAI no longer offers Completion models :/
        But below is the code (adapted to this function) that was used to obtain OpenAI Completion model results

        elif model_info['source']=='openai': 
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            SFs = [sentence + " " + followup for sentence, followup in zip(sentences, followups)]
            responses_appended = []
            n_splits = 5 # Run OpenAI API Calls in chunks, to prevent rate limit errors
            for i in range(0, len(SFs), len(SFs)//n_splits):
                SFs_slice = SFs[i:i+len(SFs)//n_splits]
                sentences_slice = sentences[i:i+len(SFs)//n_splits]
                GPT3_response_full = openai.Completion.create(engine=modelname,
                        prompt=SFs_slice,
                        max_tokens=0,
                        temperature=0.0,
                        logprobs=0,
                        echo=True,
                    )
                sentences_tokenized = tokenizer(sentences_slice)['input_ids']
                preamble_lengths = [len(tokenized_sentence) for tokenized_sentence in sentences_tokenized]
                tokenwise_logprobs = [GPT3_response_full['choices'][i]['logprobs']['token_logprobs'][preamble_lengths[i]:] for i in range(len(preamble_lengths))]
                responses = [sum(x) for x in tokenwise_logprobs]
                responses_appended = responses_appended+responses
            dataset_df[key] = responses_appended
            '''
    dataset_df.to_csv(output_file_path, index=False)
    return 

main()