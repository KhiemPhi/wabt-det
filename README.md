# Whataboutism Detection with MINA (Mining Negatives with Attention)
This is the official source code for **Detection and Dissection of Whataboutism Beyond ''Tu Quoque''**. 


## Instalation:

To run the codebase, you can install a Conda environment with the following commands:

```shell
conda create --name wabt-det python=3.8
conda activate wabt-det
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install pytorch-lightning==1.5.0
pip install transformers
pip install scikit-learn
pip install optuna
```

Alternatively, you can also run:

```shell
bash install.sh
```

## Datasets

Located in the dataset folder are the .csv files containg the YouTube comments and their annotations.

You can collect more data for further testing by in TQ-YT using the following command:

```shell
python -u dataset/collect.py --api_key [your Youtube API key] --topic [the topic of videos you want collected] --save_as [file path to save your .csv results]
```

More instructions on how to set up a YouTube API-V3 API key can be found in this following tutorial https://www.youtube.com/watch?v=th5_9woFJmk

annotations_1645.csv contains all 3 annotators annotations for TQ-YT

twitter_data_1204.csv contains all 3 annotators annotations for TQ-TW

data_split contains the train,test,val split for TQ-YT and TQ-TW in seperate .csv files.

## Context Tuples
Context Tuples are located in the context_json folder. If you would like to create your own tuples, follow the same json structure with your own data. These tuples are for our best results. Your results are also summarized in this folder, specified by the -tp flag.

## Training and Evaluation

For quick evaluation, you can run the following command: 

```shell
bash test.sh
```

This will allow you to view all test results in different text-files for the various F1-scores. The main.py script both trains the model and evalute the model on the test set each epoch. It automaticall registers the epoch with the best F1 results with the help of the PyTorch-Lightning wrappers.  Context tuples for testing are provided in the context_json folder. 

For training, you can run the commands in 

```shell
bash train.sh
```
These will train two models for TQ-TW and TQ-YT and save all the randomly generated context tuples as well as the weights. Weights are stored in best_ckpts and a table of F1 results is stored in context_json by default. 
