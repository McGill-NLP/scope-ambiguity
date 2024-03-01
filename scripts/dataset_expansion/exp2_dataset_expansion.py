import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import openai 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai-api-key', type=str)
    parser.add_argument('--source-dataset-filepath', type=str)
    parser.add_argument('--generated-data-directory', type=str)
    parser.add_argument('--n-sampled', type=int, default=5)
    parser.add_argument('--n-generated', type=int, default=10)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=3535)
    args = parser.parse_args()
    print(f"Arguments: {args}")
    generate_data(args)

# Getting type combination-wise subsets
'''
Note: this is different from the corresponding section in the exp1_dataset_expansion script.
Firstly, we have to transform the source dataframe in a way we don't have to for Experiment 1.
Secondly, we don't split by surface and inverse reading sentences, since the experiment format involves sentences that don't have strictly preferred readings.
And thirdly, due to the smaller size of the source dataset here, we only create subsets based on OP1_type, not both OP1_type and OP2_type.
'''
def pivot_df(source_df: pd.DataFrame):
    sentence_pivot = source_df.pivot_table(index='idx', columns='stype', values='sentence', aggfunc='first').reset_index()
    followup_pivot = source_df.pivot_table(index='idx', columns='ftype', values='followup', aggfunc='first').reset_index()
    
    merged_df = pd.merge(sentence_pivot, followup_pivot, on='idx', how='outer')
    
    op_cols = source_df[(source_df['stype'] == 'S') & (source_df['ftype'] == 'F1')].drop_duplicates('idx')[['idx', 'OP1', 'OP1_type', 'OP2', 'OP2_type']]
    
    transformed_df = pd.merge(merged_df, op_cols, on='idx', how='left')
    return transformed_df

def get_subsets(source_df: pd.DataFrame):
    pivoted = pivot_df(source_df)
    subsets = {}
    for OP_type in np.unique(pivoted['OP1_type']):
        subsets[OP_type] = pivoted[pivoted['OP1_type']==OP_type]

    return subsets

# Random sampling and generation
'''
This section varies between exp1_dataset_expansion and exp2_dataset_expansion due to the example format.
'''
def GPT4_generation_from_sample(sample: pd.DataFrame, n_generated: int):
    exp_preamble = "I'm trying to expand a dataset of sentences I'm using for an experiment. Below are some examples from the original dataset:\n\n"
    examples = ""
    for example in sample.iterrows():
        examples = examples+f"S: {example[1]['S']}\nSc: {example[1]['Sc']}\nF1: {example[1]['F1']}\nF2: {example[1]['F2']}\nOP1: {example[1]['OP1']}\nOP2: {example[1]['OP2']}\nOP1_type: {example[1]['OP1_type']}\nOP2_type: {example[1]['OP2_type']}\n"+"\n"
    
    n = str(n_generated)
    exp_demand = f"Now, on the basis of the sentences above, provide me another {n} similar datapoints, in a .jsonl format:"
    total_prompt = exp_preamble+examples+exp_demand
    
    GPT4_response_raw = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful model that helps build text-based datasets, but does not produce any conversation besides the text it is asked to produce."},
                {"role": "user", "content": total_prompt}],
        temperature=0.0)
    
    GPT4_response_string = GPT4_response_raw['choices'][0]['message']['content']
    
    return GPT4_response_string

# Conversion from GPT-4 output to dictionary
def jsonl_string_to_dictlist(jsonl_string):
    list_of_dicts = []
    lines = jsonl_string.split('\n')
    
    for line in lines:
        if line.strip():
            try:
                data_dict = json.loads(line.strip(","))
                list_of_dicts.append(data_dict)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(e)
    
    return list_of_dicts


def generate_data(args):
    source_dataset_filepath = args.source_dataset_filepath
    openai.api_key = args.openai_api_key
    generated_data_directory = args.generated_data_directory
    if not os.path.exists(generated_data_directory):
        os.mkdir(generated_data_directory)
    
    n_sampled = args.n_sampled
    n_generated = args.n_generated
    n_repeats = args.n_repeats
    random_seed = args.random_seed

    # Read in data, get combination-wise subsets:
    source_df = pd.read_csv(source_dataset_filepath)
    subsets = get_subsets(source_df)

    # Set seed:
    random.seed(random_seed)

    # Get data from GPT-4:
    subsets_generated = {}
    for key in tqdm(subsets.keys()):
        # Skip if .jsonl already made for this category -- hacky fix for reruns forced by API issues (can be removed if intention is to delibrately resample for completed categories)
        keystring = str(key)
        filename = f"{keystring}_GPT4_generated.jsonl"
        if filename in os.listdir(generated_data_directory): # move to next category if there's already an associated file of generations in the directory
            continue
        else:
            print(f"Starting on {keystring}")
            subsets_generated[key] = []
            for i in tqdm(range(n_repeats)):
                subset = subsets[key]
                sample = subset.sample(n_sampled, random_state=random.sample(range(0,1000),1)[0])
                # Getting raw GPT-4 output:
                generated_examples = GPT4_generation_from_sample(sample, n_generated=n_generated)
                # Converting it to dictionary:
                generated_dictlist = jsonl_string_to_dictlist(generated_examples)
                for item in generated_dictlist:
                    subsets_generated[key].append(item)
            # Write category's generated data to a .jsonl:
            full_filename = os.path.join(generated_data_directory, filename)
            with open(full_filename, 'w') as file:
                for item in subsets_generated[key]:
                    json.dump(item, file)
                    file.write("\n")
            
            print(f"Finished {keystring}!")
    
    print("All done!")
    return 


main()