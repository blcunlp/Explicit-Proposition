# Datasets and Baselines for Automatic Recognition and Analysis of Explicit Propositions

## Description
This repository provides the official training, development, and testing dataset for [Automatic Recognition and Analysis of Explicit Propositions in Natural Language](...).


## Data

The proposition-data dataset can be downloaded at [here](https://github.com/blcunlp/Explicit-Proposition/tree/master/proposition-data)

Both the train, dev, and test set are  **tab-separated** format.

### In first task(Automatic Recognition of Explicit Propositions in Natural Language): 
Each line in the train (or dev, test) file corresponds to an instance, and it is arranged as：  
>kinds-of-proposition proposition-sentence  label

### In second task(Analysis of Explicit Propositions): 
Each line in the train (or dev, test) file corresponds to an instance, and it is arranged as：  
>kinds-of-proposition proposition-sentence  labels


## Model

This repository includes the baseline models for first task and second task . 
We provide three baseline models for both tasks.

### First Task:
1. The [Support Vector Machine(SVM)Model](https://link.springer.com/content/pdf/10.1007%2FBF00994018.pdf)
2.  The Bi-directional Long Short-Term Memory(BiLSTM) Model
3.  The [BERT Model](https://arxiv.org/pdf/1810.04805.pdf), which is a strong baseline model for first task. 

### Second Task:
1.  The [Conditional Random Field(CRF)](http://www.cs.columbia.edu/~jebara/4772/papers/crf.pdf)
2.  The [BiLSTM-CRF](https://arxiv.org/pdf/1806.05626.pdf)
3.  The [BERT-BiLSTM-CRF(BBiCRF)](https://arxiv.org/pdf/1810.04805.pdf), which is a strong baseline model for first task. 

## Requirements
* python 3.5
* tensorflow      '1.13.1'

## Results 
### First Task
|Model |test-acc(%)|
|:-:|:-:|
| SVM |59.65 |
| BiLSTM |  66.80|
| BERT |  74.95 |

### Second Task
|Model |test-acc(%)| test-Precision(%) |test-Recall(%)|test-F_score(%)|
|:-:|:-:|:-:|:-:|:-:|
| CRF |73.34	|74.46|	71.70|	73.05
| BiLSTM-CRF|82.43	|78.44|	89.14	|83.45
| BERT-BiLSTM-CRF |92.04	|90.99	|90.50	|90.74

## Reporting issues
Please let us know, if you encounter any problems.
