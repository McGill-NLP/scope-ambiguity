### Experiment 1

This folder contains the script and .json file used to obtain results for Experiment 1.
  * `exp1_get_model_results.py`: script that obtains results from models specified in the `exp1_model_list.json` file, based on the model stimuli passed through in the bash script (either the Experiment 1A or 1B dataset)
  * `exp1_get_model_results.sh`: bash script used to pass arguments through to the `exp1_get_model_results.py` script
  * `exp1_model_list.json`: .json file used to specify details about models tested (change to test with other models):
    * `source`: either `"huggingface"` or `"openai"`, depending on the model
    * `type`: either `"chat"` or `"completion"`, for chat-optimized and vanilla auto-regressive models respectively
    * `model_name`: the model's official name on Huggingface or the OpenAI API
    * `quantization`: 0 or 1; whether the model should be loaded in 8-bit or not

**Note**: The scripts for obtaining model results do not currently 
