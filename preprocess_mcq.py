import os
import argparse
import spacy
from spacy.matcher import Matcher
from multiprocessing import cpu_count
import nltk
import string
import json
import re
import sys
import uuid
from multiprocessing import Pool
from tqdm import tqdm

PATTERN_PATH = './data/cpnet/matcher_patterns.json'
CPNET_VOCAB = './data/cpnet/concept.txt'
nlp = None
matcher = None
input_paths = {
    'mcq': {
        'train': './data/mcq/total_new_cleaned_train.json',
        'test': './data/mcq/total_new_cleaned_test.json',
    },
    'sciq': {
        'train': './data/sciq/train.json',
        'dev': './data/sciq/valid.json',
        'test': './data/sciq/test.json',
    },
}

output_paths = {
    'mcq': {
        'grounded':{
            'train':'./data/mcq/grounded/train.grounded.json',
            'test':'./data/mcq/grounded/test.grounded.json'
        }
    },
    'sciq': {
        'grounded':{
            'train':'./data/sciq/grounded/train.grounded.json',
            'dev': './data/sciq/grounded/valid.grounded.json',
            'test':'./data/sciq/grounded/test.grounded.json'
        }
    }
}


blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])

nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

def ground_qa_pair(qa_pair):
    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)

    s, a = qa_pair
    all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
    answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
    question_concepts = all_concepts - answer_concepts
    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    if len(answer_concepts) == 0:
        answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}

def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans=None):

    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    if ans is not None:
        ans_matcher = Matcher(nlp.vocab)
        ans_words = nlp(ans)
        # print(ans_words)
        ans_matcher.add(ans, None, [{'TEXT': token.text.lower()} for token in ans_words])

        ans_match = ans_matcher(doc)
        ans_mentions = set()
        for _, ans_start, ans_end in ans_match:
            ans_mentions.add((ans_start, ans_end))

    for match_id, start, end in matches:
        if ans is not None:
            if (start, end) in ans_mentions:
                continue

        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)

        # print("span", span)
        # print("concept", original_concept)
        # print("Matched '" + span + "' to the rule '" + string_id)

        # why do you lemmatize a mention whose len == 1?

        if len(original_concept.split("_")) == 1:
            # tag = doc[start].tag_
            # if tag in ['VBN', 'VBG']:

            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        # print("span:")
        # print(span)
        # print("concept_sorted:")
        # print(concepts_sorted)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        # print("exact match:")
        # print(exact_match)
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)

    return mentioned_concepts


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs

def hard_ground(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res

def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = []
    for item in tqdm(data):
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)

        ac = item["ac"]
        prune_ac = []
        for c in ac:
            if c[-2:] == "er" and c[:-2] in ac:
                continue
            if c[-1:] == "e" and c[:-1] in ac:
                continue
            all_stop = True
            for t in c.split("_"):
                if t not in nltk_stopwords:
                    all_stop = False
            if not all_stop and c in cpnet_vocab:
                prune_ac.append(c)

        try:
            assert len(prune_ac) > 0 and len(prune_qc) > 0
        except Exception as e:
            pass
            # print("In pruning")
            # print(prune_qc)
            # print(prune_ac)
            # print("original:")
            # print(qc)
            # print(ac)
            # print()
        item["qc"] = prune_qc
        item["ac"] = prune_ac

        prune_data.append(item)
    return prune_data




def ground(data_path, cpnet_vocab_path, pattern_path, output_path, num_processes = 1):

    # Read File
    with open(data_path, 'r') as f:
        data = json.load(f)

    sents = []
    answers = []
    distractors = []

    for d in data:

        sentence_text = d['sentence']
        distractors_list = d['distractors']
        answer_text = d['answer']

        statement = sentence_text.replace('**blank**',answer_text)
        sents.append(statement)
        answers.append(answer_text)
        distractors.append(distractors_list)

    res = []
    
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents)))
    res = prune(res, cpnet_vocab_path)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    args = parser.parse_args()

    print(f'preprocess train dataset : \n')
    ground(input_paths['mcq']['train'], CPNET_VOCAB, PATTERN_PATH, output_paths['mcq']['grounded']['train'], args.nprocs)
    print(f'preprocess test dataset : \n')
    ground(input_paths['mcq']['test'], CPNET_VOCAB, PATTERN_PATH, output_paths['mcq']['grounded']['test'], args.nprocs)
    

if __name__ == '__main__':
    main()
    # pass