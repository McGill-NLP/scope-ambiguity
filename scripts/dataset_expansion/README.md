## dataset_expansion

* `exp1_dataset_expansion.py`: script to generate data from an Experiment 1 dataset, using GPT-4.
* `exp1_dataset_expansion.py`: script to generate data from an Experiment 2 dataset, using GPT-4.
* `dataset_expansion.sh`: bash script template used to run the dataset expansion scripts and pass arguments to them.

To play with the dataset expansion used in this paper, run the python scripts in the manner shown in the `dataset_expansion.sh` file.
**Note**: if you've already run the expansion scripts once, the scripts do not currently have subsequently re-generated data overwrite already generated data. 
Be sure to either specify a new `generated-data-directory` when running the expansion scripts a second time, or delete older generated data first.
