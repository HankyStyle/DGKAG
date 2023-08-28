"""
評估Sciq實驗結果
"""

import json
import argparse
from math import log

def cala_repeat(pred_distractors, d_isRepeat):
    pred_distractors = pred_distractors[:3]
    dis_set = set(pred_distractors)
    repeat = len(pred_distractors) - len(dis_set)
    d_isRepeat[repeat] += 1

def cala_answer(answers_text, pred_distractors, d_isAnswer):
    cnt = 0
    for dis in pred_distractors:
        if dis == answers_text:
            cnt += 1
    d_isAnswer[cnt] += 1

def eval_idcg(actual, predicted, k):
    idcg = 0.
    
    ideal = [1 if pred in actual else 0 for pred in predicted]
    ideal.sort(reverse=True)
    for i in range(1, k+1):
        rel = ideal[i-1]
        idcg += rel / log(i+1, 2)
    return idcg

def eval_dcg(actual, predicted, k):
    dcg = 0.
    for i in range(1, k+1):
        rel = 0
        if predicted[i-1] in actual:
            rel = 1
        dcg += rel / log(i+1, 2)
    return dcg

def eval_ndcg(actual, predicted, k):
    ndcg = 0.
    dcg = eval_dcg(actual, predicted, k)
    idcg = eval_idcg(actual, predicted, k)
    if idcg != 0:
        ndcg += dcg / idcg
    return ndcg

def eval_map(actual, predicted):
    _map = 0.
    candidates = predicted
    val = 0.
    rank = 1
    for i in range(1, len(candidates)+1):
        if candidates[i-1] in actual:
            val += i / rank
            rank += 1
    _map += val / len(actual)
    return _map

def eval_mrr(actual, predicted):
    mrr = 0.
    candidates = predicted
    for i in range(1, len(candidates)+1):
        if candidates[i-1] in actual:
            mrr += 1.0 / i
            break
    return mrr

def eval_recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    recall = len(act_set & pred_set) / float(len(act_set))
    
    return recall

def eval_precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    prec = len(act_set & pred_set) / float(k)
    
    return prec

def eval_f1(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    prec = len(act_set & pred_set) / float(k)
    recall = len(act_set & pred_set) / float(len(act_set))

    if prec == 0 and recall == 0:
            f1 = 0
    else:
        f1 = 2 * (prec * recall / (prec + recall))
    
    return f1

def process(actual, predicted):
    diff = len(actual)-len(predicted)
    for k in range(diff):
        predicted.append("")

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def main():

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding argument
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .json files",
    )

    args = parser.parse_args()

    data = read_json(args.data_dir)

    n_question = len(data)
    f1_3 = 0.
    p1 = 0.
    p3 = 0.
    r1 = 0.
    r3 = 0.
    r10 = 0.
    f3 = 0.
    mrr = 0.
    _map = 0.
    ndcg3 = 0.
    ndcg10 = 0.
    d_isAnswer = [0, 0, 0, 0]
    d_isRepeat = [0, 0, 0]
    d_isUnique = [0, 0, 0, 0]
    isGenerated = True
    if 'score' in data[0]:
        isGenerated = False

    for _id in range(len(data)):
        d = data[_id]

        if 'distractors' not in d.keys():
            labels = [d['distractor1'],d['distractor2'],d['distractor3']]
        else:
            labels = d['distractors']

        if 'answer' not in d.keys():
            answer = d['correct_answer']
        else:
            answer = d['answer']

        #labels = [x.lower() for x in labels]
        # 避免dataset的answer有空白
        answer = answer.strip()
        
        pred_distractors = []
        if isGenerated:
            pred_distractors = d['pred_distractors']
            #pred_distractors = [x.lower() for x in pred_distractors]
        else:
            score = d['score']

        # 有些label後會有多餘的空白，會導致算分錯誤(id: 88, 241)
        labels = [item.strip() for item in labels]

        if isGenerated:

            # p_1 = eval_precision(labels, pred_distractors, 1)
            # p_3 = eval_precision(labels, pred_distractors, 3)
            # r_3 = eval_recall(labels, pred_distractors, 3)

            # if p_3 == 0 and r_3 == 0:
            #     f_3 = 0
            # else:
            #     f_3 = 2 * (p_3 * r_3 / (p_3 + r_3))

            # p1 += p_1
            # p3 += p_3
            # r3 += r_3
            # f1_3 += f_3

            # 將pred_distractors補上空白讓長度和labels一樣長避免計算時 index out of range
            process(labels, pred_distractors)



            p1 += eval_precision(labels, pred_distractors, 1)
            p3 += eval_precision(labels, pred_distractors, 3)
            r1 += eval_recall(labels, pred_distractors, 1)
            r3 += eval_recall(labels, pred_distractors, 3)
            mrr += eval_mrr(labels, pred_distractors)
            _map += eval_map(labels, pred_distractors)
            ndcg3 += eval_ndcg(labels, pred_distractors, 3)

            f1_3 += eval_f1(labels, pred_distractors, 3)

            #cala_answer(answer, pred_distractors, d_isAnswer)
            cala_repeat(pred_distractors, d_isRepeat)
        else:

            for top in range(10):
                pred_distractors.append(score[top]['word'])

            p_1 = eval_precision(labels, pred_distractors, 1)
            p_3 = eval_precision(labels, pred_distractors, 3)
            r_3 = eval_recall(labels, pred_distractors, 3)
            if p_3 == 0 and r_3 == 0:
                f_3 = 0
            else:
                f_3 = 2 * (p_3 * r_3 / (p_3 + r_3))

            p1 += p_1
            p3 += p_3
            r3 += r_3
            f1_3 += f_3
            r10 += eval_recall(labels, pred_distractors, 10)
            mrr += eval_mrr(labels, pred_distractors)
            ndcg10 += eval_ndcg(labels, pred_distractors, 10)

            cala_answer(answer, pred_distractors, d_isAnswer)
            cala_repeat(pred_distractors, d_isRepeat)

    # p1 = p1 * 100.0 / n_question
    # p3 = p3 * 100.0 / n_question
    # r3 = r3 * 100.0 / n_question
    # f1_3 = f1_3 * 100.0 / n_question
    # r10 = r10 * 100.0 / n_question
    # mrr = mrr * 100.0 / n_question
    # ndcg10 = ndcg10 * 100.0 / n_question
    p1 = p1 * 100.0 / n_question
    p3 = p3 * 100.0 / n_question
    r1 = r1 * 100.0 / n_question
    r3 = r3 * 100.0 / n_question
    f1_3 = f1_3 * 100.0/n_question
    r10 = r10 * 100.0 / n_question
    mrr = mrr * 100.0 / n_question
    _map = _map * 100.0 / n_question
    ndcg3 = ndcg3 * 100.0 / n_question
    ndcg10 = ndcg10 * 100.0 / n_question

    
    if isGenerated:
        result = {
            'P@1': p1,
            'R@1': r1,
            'P@3': p3,
            'R@3': r3,
            'MRR': mrr,
            'NDCG@3': ndcg3,
            'F1@3' : f1_3
        }
    else:
        result = {
            'P@1': p1, 
            'P@3': p3, 
            'R@3': r3,
            'F1@3': f1_3,
            'R@10': r10, 
            'MRR': mrr, 
            'NDCG@10': ndcg10
        }
    print(result)
    print('distractor is answer:')
    for k in range(len(d_isAnswer)):
        print(k, d_isAnswer[k], d_isAnswer[k] / n_question * 100)
    print()
    print('distractor is repeat:')
    for k in range(len(d_isRepeat)):
        print(k, d_isRepeat[k], d_isRepeat[k] / n_question * 100)
    print()
if __name__ == '__main__':

    main()
