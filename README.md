# ATNLP

# Individual project
This directory contains python notebooks necessary to replicate the results of Lake, B., & Baroni, M. (2018, July). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International conference on machine learning (pp. 2873-2882). PMLR. Each notebook are named with the experiment they contain, and a description is seen below:

|Name |Description|
|-----|--------|
|Exp1 |Simple split and random subset models |
|Exp2 |Length split|
|Exp3a |Models using "turn left" primitive |
|Exp3b |Models using "jump" primitive|
|Exp3c |Error inspection of experiment 3a and 3b|
|Exp3d |Cosine similarity |
|Exp3e |Model with different amount of compositional jump commands|

First, you need to download the SCAN tasks used as training and test data in this model. They can be cloned from this repository https://github.com/brendenlake/SCAN. Clone it to you desktop and rename the repository "SCAN-master". 
Open either of the notebooks and replace the os.chdir() with the directory where you have both the folder with the notebooks and the SCAN-master folder. Then, you can run the entire notebook to reproduce results.
