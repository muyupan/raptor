import re
import string
from collections import Counter
from typing import Tuple, List, Any, Dict
import json
from ollama import chat, ChatResponse

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def retrieval_accuracy(answer, contexts):
    response: ChatResponse = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 
        'content': f'You are acting as a strict evaluator. For the provided respond, check whether the Answer is either directly present, existing, or highly relevant to the Answer provided below. If yes, return ONLY one word: True, if not, return False \n No explanation, no extra text. \n Answer: {answer} \n Respond: {contexts}'}],
    )
    return (response['message']['content'])

def eval_answer(qa):
    prediction, gold, contexts = qa['respond'], qa['answer'], qa['contexts']
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)

    retrieval = 0
    retrieval_accurate = 0
    for i, context in enumerate(contexts):  
        retrieval_acc = retrieval_accuracy(gold, context)
        print(retrieval_acc)
        if retrieval_acc == "True":
            retrieval += 1
        else:
            retrieval += 0
    print(retrieval)
    print(len(contexts))
    retrieval_accurate = retrieval / len(contexts)
    return em, f1, prec, recall, retrieval_accurate

with open("/scratch1/mfp5696/250713_raptor/250723_raptor/raptor_huggingface_deepseek8b_llama7b.json") as fp:
    qa_pairs = json.load(fp)
    em = 0.
    f1 = 0.
    prec = 0.
    recall = 0.
    retrieval = 0.

    for idx, qa in enumerate(qa_pairs):
        em_s, f1_s, prec_s, recall_s, retrieval_acc = eval_answer(qa)  # Unpack all 4 values
        em += em_s
        f1 += f1_s
        prec += prec_s
        recall += recall_s
        retrieval += retrieval_acc
        print(retrieval_acc)
        print("finished 1")

    # Calculate averages
    num_questions = len(qa_pairs)
    print(f"Exact Match: {em/num_questions:.3f}")
    print(f"F1 Score: {f1/num_questions:.3f}")
    print(f"Precision: {prec/num_questions:.3f}")
    print(f"Recall: {recall/num_questions:.3f}")
    print(f"retrieval_accuracy: {retrieval/num_questions:.3f}")
