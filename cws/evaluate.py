#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate precision, recall and f1 by segment result and gold result.
passing list or file
"""

import codecs
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from dataset import word2states


def cws_eval(test_result, gold_result) -> tuple:
    if isinstance(test_result, str):
        test_result = _lines(test_result)
        gold_result = _lines(gold_result)
    return _eval_lines(test_result, gold_result)


def _lines(file: str) -> list:
    with codecs.open(file, encoding='utf8') as f:
        return f.readlines()


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def _eval_lines(test_result: list, gold_result: list) -> tuple:
    test_states = [''.join([word2states(word) for word in line.strip().split()]) for line in test_result]
    gold_states = [''.join([word2states(word) for word in line.strip().split()]) for line in gold_result]

    print(bio_classification_report(test_states, gold_states))

    success_cnt = 0
    seg_cnt = 0
    gold_cnt = sum([len(line.strip().split()) for line in gold_result])
    for test, gold, test_s, gold_s in zip(test_result, gold_result, test_states, gold_states):
        start = 0
        for word in test.strip().split():
            if sum([1 if test_s[i] == gold_s[i] else 0 for i in range(start, start + len(word))]) == len(word):
                success_cnt += 1
            start += len(word)
            seg_cnt += 1

    recall = float(success_cnt) / float(gold_cnt)
    precision = float(success_cnt) / float(seg_cnt)
    f1 = (2 * recall * precision) / (recall + precision)
    return precision, recall, f1


if __name__ == '__main__':
    # a,b,c = eval_func('../data/result/cws_hmm_result.txt','../data/testset1/test_cws1.txt')
    a, b, c = cws_eval('../data/result/cws_crf_result.txt', '../data/testset1/test_cws1.txt')
    print(a, b, c)
