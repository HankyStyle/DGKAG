# DG-KAG: Distractor Generation using Language Models with Knowledge Augmented Generation

This repo provides the source code & data of our paper: DG-KAG:Boosting Distractor Generation via Knowledge Triplet Augmentation.

<!-- This repo provides the source code & data of our paper: [DG-KAG:Boosting Distractor Generation via Knowledge Triplet Augmentation](link) (EMNLP 2023?). -->

<p align="center">
  <img src="./figs/Overview of Knowledge Augmented Generation.png" width="1000" title="Overview of DG-KAG" alt="">
</p>


## Usage

### 0. Dependencies
Run the following commands to create a project environment (assuming CUDA10.1):
```bash
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0
pip install nltk spacy==2.1.6
python -m spacy download en

# for torch-geometric
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
```


### 1. Download data
We use the distractor generation datasets (*MCQ*, *Sciq*) and the ConceptNet knowledge graph.
Download all the raw data by
```
sh download_raw_data.sh
```

Preprocess the concept raw data by running
```
python preprocess_concept.py -p <num_processes>
```
The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)


Preprocess the MCQ and Sciq data by running
```
python preprocess_mcq.py -p <num_processes>
python preprocess_mcq.py -p <num_processes>
```
The script will:
* Identify all mentioned concepts in the questions and answers

Preprocess the MCQ and Sciq data by running
```
python retrieve_mcq_triplet.py 
python retrieve_sciq_triplet.py 
```
* Extract relevant triplet for each q-a pair and candidate set
<!-- * Generate distractor candidate set for each questions and answers (Masked LM) -->


The resulting file structure will look like:

```plain
.
├── README.md
├── data/
    ├── cpnet/                 (prerocessed ConceptNet)
    ├── mcq/
        ├── train.json
        ├── valid.json
        ├── test.json
        ├── grounded/              (grounded entities)
        ├── candidate_set/         (candidate sets from candidate generator)
        └── triplets/              (extracted triplets from kg retriever)
    └── sciq/
├── modeling/                  (train model)
    ├── KAG                    (KAG model)
        ├── mcq/
            ├── t5/
            └── bart/
    └── Reranker               (Reranker model)
        ├── mcq/
        └── sciq/
├── saved_models/
├── predictions/
├── eval/                      (eval model)
    ├── eval_mcq.py
    └── eval_sciq.py
└── tutorial_material/         (some tutorial stuff)
```

### 2. Train DG-KAG
For MCQ and Sciq, run juypter notebook in modeling folder


### 3. Evaluate trained model
For MCQ and Sciq, run eval python file in eval folder
```
python eval_mcq.py --data_dir <prediction_file_path>
python eval_sciq.py --data_dir <prediction_file_path>
```


## Trained model examples
MCQ
<table>
  <tr>
    <th>Trained model</th>
    <th>Test F1@3</th>
    <th>Test NDCG@3</th>
  </tr>
  <tr>
    <th>T5 Triplet Augmentation</th>
    <th>16.47</th>
    <th>30.99</th>
  </tr>
</table>

Sciq
<table>
  <tr>
    <th>Trained model</th>
    <th>Test F1@3</th>
    <th>Test NDCG@3</th>
  </tr>
  <tr>
    <th>T5 Triplet Augmentation(with only answer triplet)</th>
    <th>16.91</th>
    <th>32.86</th>
  </tr>
</table>

**Note**: The models were trained and tested with HuggingFace transformers==3.4.0.

## Acknowledgment
This repo is built upon the following work:
```
Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering. Yanlin Feng*, Xinyue Chen*, Bill Yuchen Lin, Peifeng Wang, Jun Yan and Xiang Ren. EMNLP 2020.
https://github.com/INK-USC/MHGRN


QA-GNN: Question Answering using Language Models and Knowledge Graphs. Michihiro Yasunaga and Hongyu Ren and Antoine Bosselut and Percy Liang and Jure Leskovec. NAACL 2021.
https://github.com/michiyasunaga/qagnn
```
Many thanks to the authors and developers!