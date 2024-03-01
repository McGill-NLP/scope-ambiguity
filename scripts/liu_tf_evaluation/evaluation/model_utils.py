import torch
import os 
from pathlib import Path
# LLAMA_PATH = Path('/gscratch/scrubbed/alisaliu/llama_hf') #Commented out because I'll pull from huggingface

auth_token = os.getenv("AUTH_TOKEN") # For Llama-2 from HF
cache_dir = os.getenv("CACHE_DIR") # If separate cache_dir preferred for large models

GPT3_BATCH_SIZE = 4
FLAN_BATCH_SIZE = 16
LLAMA_BATCH_SIZE = 1


def get_model_class(model_name):
    if any([s in model_name for s in ['ada', 'babbage', 'curie', 'davinci']]):
        return 'gpt3'
    elif model_name == 'gpt-3.5-turbo' or model_name == 'gpt-4':
        return 'chat'
    elif model_name in [f'flan-t5-{size}' for size in ['small', 'base', 'large', 'xl', 'xxl']]:
        return 'flan'
    elif model_name in [f'Llama-2-{size}' for size in ['7b', '13b', '70b']]:# Minor edits to work with Llama-2 instead of Llama-1
        return 'llama'
    elif model_name in [f'Llama-2-{size}-chat' for size in ['7b', '13b', '70b']]:
        return 'llama'
    

def load_model(model_name):
    model_class = get_model_class(model_name)
    if model_class in ['gpt3', 'chat']:
        return dict()
    elif model_class == 'flan':
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f'Loading google/{model_name} from HuggingFace')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f"google/{model_name}",
            device_map='auto',
            offload_folder='offload_folder',
            torch_dtype='auto',
            offload_state_dict=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")
    elif model_class == 'llama':
        from transformers import AutoTokenizer, LlamaForCausalLM
        print(f'Loading {model_name}') # Adjusting this section so it pulls from HF and applies quantization to Llama-2 at 70-b
        if '70b' in model_name:
            model = LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}-hf",
                token=auth_token, cache_dir=cache_dir, load_in_8bit=True).eval()
        else:
            model = LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}-hf",
                token=auth_token, cache_dir=cache_dir, device_map="auto").eval()
        
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}-hf")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
    return {'model': model, 'tokenizer': tokenizer}


def is_instruct_model(model_name):
    if 'flan' in model_name or 'text' in model_name or get_model_class(model_name) == 'chat':
        return True
    return False