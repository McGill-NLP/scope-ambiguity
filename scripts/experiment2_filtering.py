import argparse
import numpy as np 
import pandas as pd 

def normalize(x, epsilon):
    return np.log((x-1+epsilon)/(6+epsilon))

def filter_dataset(args):
    epsilon = args.epsilon 
    path_to_human_results = args.path_to_human_results
    output_path = args.output_path

    human_results = pd.read_csv(path_to_human_results)
    grouped = human_results.groupby(['idx', 'stype', 'ftype'])['response'].mean()

    proxy_alphas = []
    for idx in np.unique(human_results['idx']):
        SF1_norm = normalize(grouped[(idx, 'S', 'F1')], epsilon)
        SF2_norm = normalize(grouped[(idx, 'S', 'F2')], epsilon)
        ScF1_norm = normalize(grouped[(idx, 'Sc', 'F1')], epsilon)
        ScF2_norm = normalize(grouped[(idx, 'Sc', 'F2')], epsilon)
        proxy_alpha = -((SF1_norm-SF2_norm) - (ScF1_norm-ScF2_norm))
        proxy_alphas.append((idx, proxy_alpha))

    filtered_indices = [result[0] for result in proxy_alphas if result[1]>0] # Get indices of datapoints that have proxy alpha scores greater than 0

    filtered_dataset = human_results[human_results['idx'].isin(filtered_indices)]
    filtered_dataset = filtered_dataset[['idx', 'sentence', 'followup', 'stype', 'ftype', 'OP1', 'OP1_type', 'OP2', 'OP2_type']]
    filtered_dataset = filtered_dataset.drop_duplicates().reset_index(drop=True)

    filtered_dataset.to_csv(output_path, index=False)
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-human-results', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--epsilon', type=float, default=0.01)
    args = parser.parse_args()

    filter_dataset(args)
    print(f"Dataset filtered with args: {args}")

main()