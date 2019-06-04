#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Usage """

import codecs

from model.hmm import HMM
from model.crf import CRF
from evaluate import cws_eval
from cws.crf_dataset import CWSCRFDataset
from cws.hmm_dataset import CWSHMMDataset


class D:
    def __init__(self, task, method):
        self.train = 'data/trainset/train_%s.txt' % task
        self.test = 'data/testset1/test_%s1.txt' % task
        self.val = 'data/devset/val_%s.txt' % task
        self.model = 'result/cws_model_%s.pkl' % method
        self.result = 'result/cws_result_%s.txt' % method


def train(dataset, model, train_file, model_file):
    args = dataset.train_args(train_file)
    model.train(*args, model_file)


def predict(dataset, model, model_file, test_file, result_file):
    model = model.load_model(model_file)

    with codecs.open(test_file, encoding='utf8') as f:
        std_result = f.readlines()
        raw_lines = [''.join(line.split()) for line in std_result]

    with codecs.open(result_file, 'w', encoding='utf8') as f:
        for sentence in raw_lines:
            pred = model.predict(dataset.test_args(sentence))
            f.write(dataset.decode(sentence, pred) + '\n')


def evaluate(test_file, result_file, eval_func):
    precision, recall, f1 = eval_func(result_file, test_file)
    print(precision, recall, f1)


def run(dataset, model, d):
    train(dataset, model, d.train, d.model)
    predict(dataset, model, d.model, d.test, d.result)
    evaluate(d.test, d.result, cws_eval)


def cws_hmm():
    run(CWSHMMDataset(), HMM(), D('cws', 'hmm'))


def cws_crf():
    run(CWSCRFDataset(), CRF(), D('cws', 'crf'))


if __name__ == '__main__':
    cws_crf()
    # cws_hmm()
