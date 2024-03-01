### Experiment 2

This folder contains the scripts and .json file used to obtain results for Experiment 2.
  *  `modded_scorer`: contains a lightly modified (to make the scripts used here run smoother) version of the `scorer` class from [Kanishka Misra](https://kanishka.website/)'s [minicons](https://github.com/kanishkamisra/minicons) package.
  * `exp2_get_model_results.py`: script that obtains results from models specified in the `exp2_model_list.json` file, based on the model stimuli passed through in the bash script (either the Experiment 2A or 2B dataset)
  * `exp2_get_model_results.sh`: bash script used to pass arguments through to the `exp2_get_model_results.py` script
    * Note that OUTPUT_FILE_PATH must end in .csv!
  * `exp2_model_list.json`: .json file used to specify details about models tested (change to test with other models):
    * `source`: either `"huggingface"` or `"openai"`, depending on the model
    * `model_name`: the model's official name on Huggingface or the OpenAI API
    * `quantization`: 0 or 1; whether the model should be loaded in 8-bit or not
  * `experiment2_filtering.py`: script used to filter the Experiment 2 datasets based on the results from human crowdsourced experiments
  * `experiment2_filtering.sh`: bash script used to run the `experiment2_filtering.py` script

**Note**: The scripts for obtaining model results do not currently have newly generated data overwrite previously generated data at the old OUTPUT_FILE_PATH. 
Instead, when run, the script will first read in any data already present at the OUTPUT_FILE_PATH, and only add new data for any new models that aren't present in the data at the output file location.
This was done so that in case of any errors (such as API outages) that prematurely kill the script, re-running the script will have it simply pick up where it left off, without starting over from scratch.
But if you've already run the script once and would like to re-generate the same data, make sure to either specify a different OUTPUT_FILE_PATH, or delete the previously generated data first. 
