import os 
import pandas as pd 
import numpy as np 
import scipy.stats as stats

#############################################################################################################
############################################## Experiment 1 #################################################
#############################################################################################################

def get_human_analysis(path_to_human_results):
    human_results_df = pd.read_csv(path_to_human_results)
    human_acc = np.mean(human_results_df['response']==human_results_df['gold_ans'])
    human_surface = human_results_df[human_results_df['gold_scope_label']=='surface']
    human_acc_surface = np.mean(human_surface['response']==human_surface['gold_ans'])
    human_inverse = human_results_df[human_results_df['gold_scope_label']=='inverse']
    human_acc_inverse = np.mean(human_inverse['response']==human_inverse['gold_ans'])
    print(f'Humans:\nAcc: {human_acc}\nAcc on Surface Reading Sentences: {human_acc_surface}\nAcc on Inverse Reading Sentences: {human_acc_inverse}\n\n\n')
    return 

'''
The experiment 1A datasets are copyright-protected, and therefore not included in the repository.
See the README for how to access the datasets.

path_to_exp1a_human = "human_results/exp1a_final_human_results_cleaned.csv"
get_human_analysis(path_to_exp1a_human)
'''

def get_model_analyses(path_to_model_results):
    '''
    This function is a little hacky as a result of the variation in model responses.
    Manual inspection of model results showed that while for completion models (both OpenAI and Llama-2), responses were easy to evaluate, 
    chat models provided less consistent responses, and were harder to evaluate.
    So we treat responses differently based on which model they came from:
    '''
    model_results_df = pd.read_csv(path_to_model_results)
    
    completion_model_names = ['Llama-2 7B', 'Llama-2 7B Control', 'Llama-2 13B', 'Llama-2 13B Control', 'Llama-2 70B', 'Llama-2 70B Control',
                                'GPT-3 davinci', 'GPT-3 davinci Control', 'GPT-3.5 td-002', 'GPT-3.5 td-002 Control', 'GPT-3.5 td-003', 'GPT-3.5 td-003 Control']
    
    openai_chat_model_names = ['GPT-3.5 Turbo', 'GPT-3.5 Turbo Control', 'GPT-4', 'GPT-4 Control']
    llama2_chat_model_names = ['Llama-2 7B Chat', 'Llama-2 7B Chat Control', 'Llama-2 13B Chat', 'Llama-2 13B Chat Control', 'Llama-2 70B Chat', 'Llama-2 70B Chat Control']
    chat_model_names = openai_chat_model_names+llama2_chat_model_names
    
    ### For Chat Models: 
    # Formatting of responses is 'Option X' instead of just 'X'
    conversion = {'Option A': 'A', 'Option B': 'B'} # dict to convert model responses to gold answers format 
    
    # Token overlap for extracting answers (e.g. 'Option B: All d' => 'Option B' treated as response): 
    # (We do this since model responses not completely consistent in format)
    def get_answer(model_response: str):
        if 'Option A' in model_response:
            return 'Option A'
        elif 'Option B' in model_response:
            return 'Option B'
        else:
            return 'misc'
    
    for model_name in chat_model_names+completion_model_names:
        if model_name in completion_model_names:
            acc = np.mean(model_results_df[model_name]==model_results_df['gold_ans'])
            surface = model_results_df[model_results_df['gold_scope_label']=='surface']
            acc_surface = np.mean(surface[model_name]==surface['gold_ans'])
            inverse = model_results_df[model_results_df['gold_scope_label']=='inverse']
            acc_inverse = np.mean(inverse[model_name]==inverse['gold_ans'])
            print(f'{model_name}:\nAcc: {acc}\nAcc on Surface Reading Sentences: {acc_surface}\nAcc on Inverse Reading Sentences: {acc_inverse}\n\n\n') 
        
        elif model_name in chat_model_names:
            answers_formatted = model_results_df[model_name].apply(lambda x: get_answer(x)).replace(conversion)
            acc = np.mean(answers_formatted==model_results_df['gold_ans'])
            surface = model_results_df[model_results_df['gold_scope_label']=='surface']
            surface_answers = answers_formatted[model_results_df['gold_scope_label']=='surface']
            acc_surface = np.mean(surface_answers==surface['gold_ans'])
            inverse = model_results_df[model_results_df['gold_scope_label']=='inverse']
            inverse_answers = answers_formatted[model_results_df['gold_scope_label']=='inverse']
            acc_inverse = np.mean(inverse_answers==inverse['gold_ans'])
            print(f'{model_name}:\nAcc: {acc}\nAcc on Surface Reading Sentences: {acc_surface}\nAcc on Inverse Reading Sentences: {acc_inverse}\n\n\n') 
    #
    return 


# Experiment 1A:
'''
The experiment 1A datasets are copyright-protected, and therefore not included in the repository.
See the README for how to access the datasets.

path_to_exp1a_model_results = "model_results/exp1a_model_results.csv"
get_model_analyses(path_to_exp1a_model_results)
'''

# Experiment 1B:
path_to_exp1b_model_results = "model_results/exp1b_model_results.csv"
get_model_analyses(path_to_exp1b_model_results)


#############################################################################################################
############################################## Experiment 2 #################################################
#############################################################################################################

def get_human_proxy_scores(human_results_df):
    # Get means of each datapoint:
    human_results_df = human_results_df.groupby(['idx', 'stype', 'ftype', 'sentence', 'followup', 'OP1', 'OP2', 'OP1_type', 'OP2_type'])['response'].mean().reset_index()
    
    # Normalize and then get proxy alpha scores:
    def normalize(x, e=0.01):
        return np.log((x-1+e)/(6+e))
    human_results_df['norm'] = human_results_df['response'].apply(lambda x: normalize(x))
    
    # Get scores for S+F1, S+F2, Sc+F1, Sc+F2 configurations:
    SF1 = human_results_df[(human_results_df['stype']=='S') & (human_results_df['ftype']=='F1')]['norm'].to_numpy()
    SF2 = human_results_df[(human_results_df['stype']=='S') & (human_results_df['ftype']=='F2')]['norm'].to_numpy()
    ScF1 = human_results_df[(human_results_df['stype']=='Sc') & (human_results_df['ftype']=='F1')]['norm'].to_numpy()
    ScF2 = human_results_df[(human_results_df['stype']=='Sc') & (human_results_df['ftype']=='F2')]['norm'].to_numpy()
    
    # Get differences of log probs:
    S_diffs = SF1 - SF2
    Sc_diffs = ScF1 - ScF2
    
    # Get p-value from paired t-test:
    p_val = stats.ttest_rel(S_diffs, Sc_diffs).pvalue
    
    # Get proxy alpha score:
    diff_diffs = S_diffs - Sc_diffs
    proxy_alpha = -(diff_diffs)
    
    return proxy_alpha, p_val

def get_model_analyses(model_results_df, human_alphas):
    model_names = [column_name for column_name in model_results_df.columns if column_name not in ['idx', 'sentence', 'followup', 'stype', 'ftype', 'OP1', 'OP1_type', 'OP2', 'OP2_type']]
    for model_name in model_names:
        # Get scores for S+F1, S+F2, Sc+F1, Sc+F2 configurations:
        SF1 = model_results_df[(model_results_df['stype']=='S') & (model_results_df['ftype']=='F1')][model_name].to_numpy()
        SF2 = model_results_df[(model_results_df['stype']=='S') & (model_results_df['ftype']=='F2')][model_name].to_numpy()
        ScF1 = model_results_df[(model_results_df['stype']=='Sc') & (model_results_df['ftype']=='F1')][model_name].to_numpy()
        ScF2 = model_results_df[(model_results_df['stype']=='Sc') & (model_results_df['ftype']=='F2')][model_name].to_numpy()
        
        # Get differences of log probs:
        S_diffs = SF1 - SF2
        Sc_diffs = ScF1 - ScF2 
        
        # Get p-value from paired t-test:
        p_val = stats.ttest_rel(S_diffs, Sc_diffs).pvalue
        
        # Get alpha score:
        diff_diffs = S_diffs - Sc_diffs 
        alphas = -(diff_diffs)
        
        # Get correlation with human scores:
        corr = stats.pearsonr(alphas, human_alphas)
        
        print(f'{model_name}:\nMean Alpha: {np.mean(alphas)}\np-value: {p_val}\nProp. of positive alphas: {np.mean(alphas>0)}\nPearson Corr. with Human Results: {corr}\n\n')
    return 

def present_exp2_analyses(path_to_model_results, path_to_human_results):
    exp_model = pd.read_csv(path_to_model_results)
    exp_human = pd.read_csv(path_to_human_results)
    
    # Only compare datapoints that survived filtering and ended up being model stimuli:
    exp_human = exp_human[exp_human['idx'].isin(np.unique(exp_model['idx']))]
    
    # Get, present human results:
    exp_proxy_alphas, exp_human_p_val = get_human_proxy_scores(exp_human)
    print(f"Mean of human proxy alpha scores: {np.mean(exp_proxy_alphas)}")
    print(f"p-value from paired t-test: {exp_human_p_val}")
    
    # Get, present model results:
    get_model_analyses(exp_model, exp_proxy_alphas)
    
    return


# Experiment 2A:
path_to_exp2a_human_results = "human_results/exp2a_human_results_cleaned.csv"
path_to_exp2a_model_results = "model_results/exp2a_model_results.csv"
present_exp2_analyses(path_to_exp2a_model_results, path_to_exp2a_human_results)

# Experiment 2B:
path_to_exp2b_human_results = "human_results/exp2b_human_results_cleaned.csv"
path_to_exp2b_model_results = "model_results/exp2b_model_results.csv"
present_exp2_analyses(path_to_exp2b_model_results, path_to_exp2b_human_results)


#############################################################################################################
###################################### Liu et al. Follow-up #################################################
#############################################################################################################

path_to_TF_results = "scripts/liu_et_al_tf_evaluation/TF_evaluation_data"
results_files = np.sort(os.listdir(path_to_TF_results))

for filename in results_files:
    model_name = filename[:-12] # Name minus characters for date and .jsonl filetype
    data_df = pd.read_json(os.path.join(path_to_TF_results, filename), lines=True)
    acc = np.mean(data_df['prediction']==data_df['answer'])
    TF_prob_mass = np.mean(data_df['TF_prob_mass'])
    print(f"{model_name}:\nAcc: {acc}\nAverage TF prob mass: {TF_prob_mass}\n")


