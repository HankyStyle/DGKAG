import os
import argparse
import spacy
import numpy as np
from spacy.matcher import Matcher
from multiprocessing import cpu_count
import nltk
from utils.conceptnet import merged_relations
import string
import json
import re
import sys
import uuid
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm

PATTERN_PATH = './data/cpnet/matcher_patterns.json'
grounded_path = './data/mcq/grounded/train.grounded.json'
cpnet_graph_path = './data/cpnet/conceptnet.en.pruned.graph'
cpnet_vocab_path = './data/cpnet/concept.txt'

from transformers import BertTokenizer, BertForMaskedLM, pipeline

tokenizer = BertTokenizer.from_pretrained("AndyChiang/cdgp-csg-scibert-dgen")
masked_lm = BertForMaskedLM.from_pretrained("AndyChiang/cdgp-csg-scibert-dgen")
unmasker = pipeline("fill-mask", tokenizer=tokenizer, model=masked_lm, top_k=15)


global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

input_paths = {
    'mcq': {
        'grounded':{
            'train':'./data/mcq/grounded/train.grounded.json',
            'test':'./data/mcq/grounded/test.grounded.json'
        },
        'candidate_set':{
            'generate_lm':{
                'train':'./data/mcq/candidate_set/generate_lm/train.candidate.json',
                'test':'./data/mcq/candidate_set/generate_lm/test.candidate.json'
            }
        }
    },
    'sciq': {
        'grounded':{
            'train':'./data/sciq/grounded/train.grounded.json',
            'dev': './data/sciq/grounded/valid.grounded.json',
            'test':'./data/sciq/grounded/test.grounded.json'
        },
        'candidate_set':{
            'generate_lm':{
                'train':'./data/sciq/candidate_set/generate_lm/train.candidate.json',
                'dev': './data/sciq/candidate_set/generate_lm/valid.candidate.json',
                'test':'./data/sciq/candidate_set/generate_lm/test.candidate.json'
            }
        }
    }
}

output_paths = {
    'mcq': {
        'triplets':{
            'generate_lm':{
                'train':'./data/mcq/triplets/generate_lm/train.triplet.json',
                'test':'./data/mcq/triplets/generate_lm/test.triplet.json'
            },
            'masked_lm':{
                'train':'./data/mcq/triplets/masked_lm/train.triplet.json',
                'test':'./data/mcq/triplets/masked_lm/test.triplet.json'
            },
            'generate_masked_lm':{
                'train':'./data/mcq/triplets/generate_masked_lm/train.triplet.json',
                'test':'./data/mcq/triplets/generate_masked_lm/test.triplet.json'
            },
        }
    },
    'sciq': {
        'triplets':{
            'generate_lm':{
                'train':'./data/sciq/triplets/generate_lm/train.triplet.json',
                'dev': './data/sciq/triplets/generate_lm/valid.triplet.json',
                'test':'./data/sciq/triplets/generate_lm/test.triplet.json'
            },
            'masked_lm':{
                'train':'./data/sciq/triplets/masked_lm/train.triplet.json',
                'dev': './data/sciq/triplets/masked_lm/valid.triplet.json',
                'test':'./data/sciq/triplets/masked_lm/test.triplet.json'
            },
            'generate_masked_lm':{
                'train':'./data/sciq/triplets/generate_masked_lm/train.triplet.json',
                'dev': './data/sciq/triplets/generate_masked_lm/valid.triplet.json',
                'test':'./data/sciq/triplets/generate_masked_lm/test.triplet.json'
            }
        }
    }
}

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

# 載入 concept 的 graph
def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

def read_candidate_data(candidate_path):
    with open(candidate_path) as f:
        candidate_data = json.load(f)
    return candidate_data

# 從 KG 內找出 Candidate Set 的 Entity
def extract_candidate_entity(data):
    qc_ids = data[0]
    ac_ids = data[1]
    question = data[2]
    distractors_set = data[3:]
    candidate_entity = []
    for distractor in distractors_set:
        if distractor in concept2id:
            entity_nodes_ids = concept2id[distractor] 
            candidate_entity.append(entity_nodes_ids)
    return (sorted(qc_ids), sorted(ac_ids), question, candidate_entity)

# 找出 node ids 內彼此有交集的 triplet
def find_union_triplet_between_nodes(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    triplets = []
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        triplets.append([int(e_attr['rel']), int(s_c), int(t_c), e_attr['weight']])    
    return triplets


def extract_triplet(data):
    qc_ids, ac_ids, question, candidate_ids = data
    schema_graph = qc_ids + ac_ids + candidate_ids # <== 考慮全部的組合
    # schema_graph = qc_ids + candidate_ids # <== 考慮 qc_ids 跟 candidate_ids 的組合 (without_ans)
    triplets = find_union_triplet_between_nodes(schema_graph)
    return {'triplets': triplets}

def append_extra_node_use_LM(data):
    qc_ids, ac_ids, question = data
    extra_nodes = []
    cid2score  = []
    distractors_set = unmasker(question)

    # 代表 question 的句子 有 2 個以上 的 [Mask]
    if len(distractors_set) < 5:
        for each_distractor_set in distractors_set:
            for distractor in each_distractor_set:
                if distractor['token_str'] != (question.split(' '))[-1] and distractor['token_str'] in concept2id:
                    extra_nodes_ids = concept2id[distractor['token_str']]
                    extra_nodes.append(extra_nodes_ids)
    else:

        for distractor in distractors_set:
            if distractor['token_str'] != (question.split(' '))[-1] and distractor['token_str'] in concept2id:
                extra_nodes_ids = concept2id[distractor['token_str']]
                extra_nodes.append(extra_nodes_ids)
    extra_nodes = set(extra_nodes)
    return (sorted(qc_ids), sorted(ac_ids), question, extra_nodes)

def retrieve_triplet_from_KG_generative_lm(grounded_path, candidate_path, output_path):
    # 載入 concept2id, id2relation, relation2id
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            sentence = dic['sent']
            QAcontext = "{}.[SEP] {}.".format(sentence, dic['ans'])
            qa_data.append((q_ids, a_ids, QAcontext))
    
    candidate_data = read_candidate_data(candidate_path)

    for i in range(len(qa_data)):
        candidate = candidate_data[i]['candidate_set']
        pred = []
        for words in candidate:
            words = str(words)
            for token in words.split(' '):
                if token != '' or token != ' ' and len(token) != 0:
                    pred.append(token.lower())
        qa_data[i] = qa_data[i] + tuple(pred)

    print('總共有',len(qa_data),'筆')


    res1 = list(tqdm(map(extract_candidate_entity, qa_data), total=len(qa_data)))

    res2 = list(tqdm(map(extract_triplet, res1), total=len(res1)))

    res3 = []

    for item in tqdm(res2):
        temp_list = []
        for triplets in item['triplets']:
            rel, source_node, target_node, weight = triplets
            relation = id2relation[rel]
            source = id2concept[source_node]
            target = id2concept[target_node]
            temp_list.append([relation, source, target, weight])
        res3.append(temp_list)

    # Remove Duplicate Triplet
    for i in range(len(res3)):
        res3[i] = [list(t) for t in set(tuple(element) for element in res3[i])]
    
    with open(output_path, 'w') as fout:
        json.dump(res3, fout)
    print(f'data saved to {output_path}')

def concepts2adj(question_ids,candidate_ids):
    global id2relation
    n_rel = len(id2relation)
    triplets = []
    a_ids = question_ids[-1]

    # type 2 考慮 qa_context 與 各個 distractor 的關係
    for q_ids in question_ids:
        for d_ids in candidate_ids:
            if cpnet.has_edge(q_ids, d_ids) and q_ids != d_ids:
                 for e_attr in cpnet[q_ids][d_ids].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel and e_attr['weight'] >= 1.0:
                        triplets.append([int(e_attr['rel']), int(q_ids), int(d_ids), e_attr['weight']])

    
    for d_ids in candidate_ids:
        for q_ids in question_ids:
            if cpnet.has_edge(d_ids, q_ids) and q_ids != d_ids:
                for e_attr in cpnet[d_ids][q_ids].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel and e_attr['weight'] >= 1.0:
                        triplets.append([int(e_attr['rel']), int(d_ids), int(q_ids), e_attr['weight']])

    return triplets

def concepts_to_adj_matrices_1hop_all_pair__use_LM(data):
    qc_ids, ac_ids, question, extra_nodes_ids = data
    triplets = concepts2adj(qc_ids + ac_ids,extra_nodes_ids)
    return {'triplets': triplets}

def retrieve_triplet_from_KG_masked_lm(grounded_path, output_path):
    # 載入 concept2id, id2relation, relation2id
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)


    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids

            # 有些資料不乾淨，sentence 沒 answer
            if '*blank**' in dic['sent']:
                sentence = dic['sent'].replace('*blank**','[MASK]')
            elif '**blank*' in dic['sent']:
                sentence = dic['sent'].replace('**blank*','[MASK]')
            else:
                sentence = dic['sent'].replace(dic['ans'],'[MASK]')
            QAcontext = "{}.[SEP] {}.".format(sentence, dic['ans'])
            qa_data.append((q_ids, a_ids, QAcontext))
    print('總共有',len(qa_data),'筆')
    print()
    res1 = list(tqdm(map(append_extra_node_use_LM, qa_data), total=len(qa_data)))
    res2 = list(tqdm(map(concepts_to_adj_matrices_1hop_all_pair__use_LM, res1), total=len(res1)))
    
    res3 = []

    for item in tqdm(res2):
        temp_list = []
        for triplets in item['triplets']:
            rel, source_node, target_node, weight = triplets
            relation = id2relation[rel]
            source = id2concept[source_node]
            target = id2concept[target_node]
            temp_list.append([relation, source, target, weight])
        res3.append(temp_list)
    
    with open(output_path, 'w') as fout:
        json.dump(res3, fout)
    print(f'data saved to {output_path}')
    
def main():
    # MCQ Retrieving
    print(f'retrieving train dataset triplet (generate_lm) from KG: \n')
    retrieve_triplet_from_KG_generative_lm(input_paths['mcq']['grounded']['train'], input_paths['mcq']['candidate_set']['generate_lm']['train'], output_paths['mcq']['triplets']['generate_lm']['train'])
    print(f'retrieving test dataset triplet (generate_lm) from KG: \n')
    retrieve_triplet_from_KG_generative_lm(input_paths['mcq']['grounded']['test'], input_paths['mcq']['candidate_set']['generate_lm']['test'], output_paths['mcq']['triplets']['generate_lm']['test'])

    print(f'retrieving train dataset triplet (masked_lm) from KG: \n')
    retrieve_triplet_from_KG_masked_lm(input_paths['mcq']['grounded']['train'], output_paths['mcq']['triplets']['masked_lm']['train'])
    print(f'retrieving test dataset triplet (masked_lm) from KG: \n')
    retrieve_triplet_from_KG_masked_lm(input_paths['mcq']['grounded']['test'], output_paths['mcq']['triplets']['masked_lm']['test'])

    # Sciq Retrieving

if __name__ == '__main__':
    main()
    # pass
