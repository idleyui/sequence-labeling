#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""running script for cws """

import codecs

from crf_dataset import CRFDataset
from hmm_dataset import HMMDataset
from evaluate import eval_func
from model.hmm import HMM
from model.crf import CRF


class CWS:
    def __init__(self, method):
        self.method = method
        if method == 'hmm':
            self.Model = HMM
            self.Dataset = HMMDataset
        elif method == 'crf':
            self.Model = CRF
            self.Dataset = CRFDataset
        elif method == 'bi-lstm-crf':
            pass
        else:
            raise Exception('unknown training method')

    def train(self, train_file, model_file):
        dataset = self.Dataset()
        args = dataset.train_args(train_file)

        model = self.Model()
        model.train(*args, model_file)

    def segment(self, model_file, test_file, result_file):
        model = self.Model.load_model(model_file)
        dataset = self.Dataset()

        with codecs.open(test_file, encoding='utf8') as f:
            std_result = f.readlines()
            raw_lines = [''.join(line.split()) for line in std_result]

        result_map = {
            'S': lambda w: w + ' ',
            'B': lambda w: w,
            'M': lambda w: w,
            'E': lambda w: w + ' '
        }

        with codecs.open(result_file, 'w', encoding='utf8') as f:
            for sentence in raw_lines:
                f.write(''.join(
                    [result_map[tag](word) for word, tag in zip(sentence, model.predict(dataset.test_args(sentence)))]
                ).strip() + '\n')

        # std_result = [sentence.replace('  ', ' ').strip() for sentence in std_result if sentence]

    def evaluate(self, test_file, result_file):
        precision, recall, f1 = eval_func(result_file, test_file)
        print(precision, recall, f1)
