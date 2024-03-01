### [Liu et al. 2023](https://arxiv.org/abs/2304.14399) TF Evaluation

This folder contains data and code used to compare the methods we present with one of the methods presented in [Liu et al. 2023](https://arxiv.org/abs/2304.14399) (see Section 4.2).
Much of the code here is only minorly adapted from the [original repository](https://github.com/alisawuffles/ambient) for that paper. 

  * `TF_evaluation_data`: model results from running the authors' methods on a formatted subset of our data (`liu_scope_evaluation.jsonl`)
  * `evaluation` and `generation`: folders containing modified versions (suited to our models and code) of utils required for the authors' code
  * `liu_scope_evaluation.jsonl`: random subset of our Experiment 2 data adapted to suit the authors' experimental format; contains ambiguous sentences with possible disambiguations
  * `liu_tf_evaluation.py`: script used to obtain model results in the experimental set-up presented in Section 4.2 of the authors' paper; minorly edited version of the authors' original script
  * `liu_tf_evaluation.sh`: bash script used to run the `liu_tf_evaluation.py` script.
