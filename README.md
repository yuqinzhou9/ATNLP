# ATNLP

# Individual project
The results are available at https://wandb.ai/yuqinzhou/ATNLP?workspace=user-yuqinzhou. Here are the name conventions:
"simple_split_hyper" is the hyperparameter search for exp1 using the transformer  

"len_split_hyper" is the hyperparameter search for exp2 using the transformer  

For the name like this "Exp1_var02_overall_0", it means the first run using the overall best models trained based on the 2% proportion of the Exp1 datasets.  

To reproduce the results, the "transformer" folder contains:

|Name |Description|
|-----|--------|
|ATNLP_transformer_hyper | Hyperparameter search for exp1 and exp2 using the transformer |
|ATNLP_simple_32p|Exmaple code of the overall best model trained based on the 32% proportion of the Exp1 |
|transformer_pos | transformer models with fixed positional embeddings |
|transformer_new | transformer models with learnable positional embeddings |



# Group project
The "Reimplementation" folder contains:

|Name |Description|
|-----|--------|
|Exp1 |Simple split and random subset models |
|Exp2 |Length split|
|Exp3a |Models using "turn left" primitive |
|Exp3b |Models using "jump" primitive|
|Exp3c |Error inspection of experiment 3a and 3b|
|Exp3d |Cosine similarity |
|Exp3e |Model with different amount of compositional jump commands|
