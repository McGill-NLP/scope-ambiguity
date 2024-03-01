import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd 
import openai
from tqdm import tqdm
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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


def get_completion_prompts(qa_df):
    description = "On the basis of this phrase/statement alone, and with no further context, there are only two options:\n"
    q_prompt = "The most likely option is option"
    test_prompts=[]
    for i in range(len(qa_df)):
        sentence = qa_df['sentence'].iloc[i].strip()
        options = f"Option A: {qa_df['Option A'].iloc[i]}.\nOption B: {qa_df['Option B'].iloc[i]}.\n"
        full_prompt = sentence+"\n"+description+options+q_prompt
        test_prompts.append(full_prompt)
    
    control_prompts = []
    for i in range(len(qa_df)):
        options = f"Option A: {qa_df['Option A'].iloc[i]}.\nOption B: {qa_df['Option B'].iloc[i]}.\n"
        full_prompt = description+options+q_prompt
        control_prompts.append(full_prompt)
    
    return {'test': test_prompts, 'control': control_prompts}

def get_chat_prompts(qa_df):
    description = "On the basis of this phrase/statement alone, and with no further context, there are only two options:\n"
    chat_question_prompt = "Which of the two options is most likely? Answer with either 'Option A' (for Option A) or 'Option B' (for option B)."
    test_prompts = []
    for i in range(len(qa_df)):
        sentence = qa_df['sentence'][i].strip()
        options = f"Option A: {qa_df['Option A'][i]}.\nOption B: {qa_df['Option B'][i]}.\n"
        full_prompt = sentence+"\n"+description+options+chat_question_prompt
        test_prompts.append(full_prompt)

    control_prompts = []
    for i in range(len(qa_df)):
        options = f"Option A: {qa_df['Option A'].iloc[i]}.\nOption B: {qa_df['Option B'].iloc[i]}.\n"
        full_prompt = description+options+chat_question_prompt
        control_prompts.append(full_prompt)
    
    return {'test': test_prompts, 'control': control_prompts}

def get_hf_completion_responses(model, tokenizer, prompts):
    outputs = []
    for prompt in tqdm(prompts):
        model_input = tokenizer(prompt, return_tensors="pt")
        model_input.to('cuda')
        generation_config = GenerationConfig(max_new_tokens=1, do_sample=False)
        output = model.generate(**model_input, generation_config=generation_config)
        outputs.append(output)
    
    outputs_processed = tokenizer.batch_decode([x[0][-1] for x in outputs])
    return outputs_processed
 
def get_hf_chat_responses(model, tokenizer, prompts, n_tokens=5):
    sys_prompt = "You are a helpful assistant who answers common-sense reasoning questions. Respond with only two words."
    sys_llama = f"<<S>> {sys_prompt} <</SYS>>"
    llama_formatted_prompts = [f"{sys_llama}\n[INST] {prompt} [/INST]\n" for prompt in prompts]
    outputs=[]
    for prompt in tqdm(llama_formatted_prompts):
        model_input = tokenizer(prompt, return_tensors="pt")
        model_input.to('cuda')
        generation_config = GenerationConfig(max_new_tokens=n_tokens, do_sample=False)
        output = model.generate(**model_input, generation_config=generation_config)
        outputs.append(output)
    
    outputs_processed = tokenizer.batch_decode([x[0][-n_tokens:] for x in outputs])
    return outputs_processed
    
def get_openai_chat_responses(model_name, prompts, n_tokens=2):
    checkpoint_filepath = ".openai_chat_checkpoint" # Hacky checkpointing to save progress on dataset if API calls fail somewhere in the middle of a dataset
    if not os.path.exists(checkpoint_filepath):
        state = {'checkpoint': 0, 'outputs': []}
    else:
        with open(checkpoint_filepath, 'rb') as file:
            state = pickle.load(file)
    
    checkpoint, outputs = state['checkpoint'], state['outputs']
    sys_prompt = "You are a helpful assistant who answers common-sense reasoning questions. Respond with only two words."
    for prompt in tqdm(prompts[checkpoint:]):
        response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2
        )
        checkpoint += 1
        outputs.append(response)
        with open(checkpoint_filepath, 'wb') as file:
            pickle.dump({'checkpoint': checkpoint, 'outputs': outputs}, file)
    
    outputs_processed = [x['choices'][0]['message']['content'].strip() for x in outputs]

    with open(checkpoint_filepath, 'wb') as file:
        pickle.dump({'checkpoint': 0, 'outputs': []}, file)
        
    return outputs_processed


def get_model_results(args):
    path_to_model_stimuli = args.path_to_model_stimuli
    path_to_model_list = args.path_to_model_list
    output_file_path = args.output_file_path
    openai.api_key = args.openai_api_key
    auth_token = args.auth_token # for gated models like Llama 2
    cache_dir = args.cache_dir # if specific cache directory required for disk space limit reasons

    qa_df = pd.read_csv(path_to_model_stimuli)
    if os.path.exists(output_file_path):
        qa_df = pd.read_csv(output_file_path) # Hacky fix for restarting the script if it's failed because of API problems -- re-start on a half-complete output_file_path.csv file
    completion = get_completion_prompts(qa_df)
    chat = get_chat_prompts(qa_df)
    completion_test = completion['test']
    completion_control = completion['control']
    chat_test = chat['test']
    chat_control = chat['control']

    with open(path_to_model_list, 'r') as file:
        data = file.read()
        model_list = json.loads(data)

    for key in model_list.keys():
        model_info = model_list[key]
        model_name = model_info['model_name']
        quantization_bool = bool(model_info['quantization'])
        if (str(key) in qa_df.columns) and (str(key)+" Control" in qa_df.columns): # See comment on line 114 -- for restarting on a half-complete output_file_path.csv file
            continue
        
        if model_info['type']=='completion':
            if model_info['source']=='huggingface':
                print(f"Starting on {key}:")
                if quantization_bool:
                    model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir, load_in_8bit=True).eval()
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir).eval()
                    model.to('cuda')
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir)
                print(f"Successfully imported {model_name}!")
                # Test prompts:
                if str(key) not in qa_df.columns:
                    print("Starting on test prompts...")
                    test_responses = get_hf_completion_responses(model, tokenizer, completion_test)
                    qa_df[key] = test_responses
                    qa_df.to_csv(output_file_path, index=False)
                    
                # Control prompts:
                if str(key)+" Control" not in qa_df.columns:
                    print("Starting on control prompts...")
                    control_responses = get_hf_completion_responses(model, tokenizer, completion_control)
                    qa_df[str(key)+" Control"] = control_responses
                    qa_df.to_csv(output_file_path, index=False)
                
                del(model)
                gc.collect()
                torch.cuda.empty_cache()
            
            if model_info['source']=='openai':
                continue
                '''
                Deprecated! Since OpenAI no longer offers Completion models :/
                But below is a version of the code (adapted to this function) that was used to obtain OpenAI Completion model results
                
                print(f'Starting on {str(key)}')
                # Test responses:
                print("Starting on test prompts...")
                test_responses = openai.Completion.create(engine=model_name, prompt=completion_test, max_tokens=1, temperature=0.0)
                answers = pd.Series([x['text'].strip() for x in responses['choices']])
                qa_df[key]=answers
                print("Starting on control prompts...")
                control_responses = openai.Completion.create(engine=model, prompt=completion_control, max_tokens=1, temperature=0.0) # Get GPT-3/3.5 responses
                control_answers = pd.Series([x['text'].strip() for x in control_responses['choices']])
                qa_df[str(key)+ ' Control'] = control_answers
                qa_df.to_csv(output_file_path, index=False)
                print(f"{str(key)} done!")
                '''
            
        if model_info['type']=='chat':
            if model_info['source']=='huggingface':
                print(f"Starting on {key}:")
                if quantization_bool:
                    model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir, load_in_8bit=True).eval()
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir).eval()
                    model.to('cuda')
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, cache_dir=cache_dir)
                print(f"Successfully imported {model_name}!")
                # Test prompts:
                if str(key) not in qa_df.columns:
                    print("Starting on test prompts...")
                    test_responses = get_hf_chat_responses(model, tokenizer, chat_test)
                    qa_df[key] = test_responses
                    qa_df.to_csv(output_file_path, index=False)
                
                # Control prompts:
                if str(key)+" Control" not in qa_df.columns:
                    print("Starting on control prompts...")
                    control_responses = get_hf_chat_responses(model, tokenizer, chat_control)
                    qa_df[str(key)+" Control"] = control_responses
                    qa_df.to_csv(output_file_path, index=False)
                
                del(model)
                gc.collect()
                torch.cuda.empty_cache()
            
            if model_info['source']=='openai':
                print(f"Starting on {key}:")
                # Test responses:
                if str(key) not in qa_df.columns:
                    print(f"Starting on test prompts...")
                    test_responses = get_openai_chat_responses(model_name, chat_test)
                    qa_df[key] = test_responses
                    qa_df.to_csv(output_file_path, index=False)
                # Control responses:
                if str(key)+" Control" not in qa_df.columns:
                    print(f"Starting on control prompts...")
                    control_responses = get_openai_chat_responses(model_name, chat_control)
                    qa_df[str(key)+" Control"] = control_responses
                    qa_df.to_csv(output_file_path, index=False)
            
    print("All done!")
    return 

main()