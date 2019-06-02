#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""running script for cws """

import argparse
import codecs

from hmm_model import Model as HMMModel
from crf_model import CRF as CRFModel
from dataset import Dataset
from evaluate import eval_func


class CWS:
    def __init__(self, method):
        if method == 'hmm':
            self.Model = HMMModel
        elif method == 'crf':
            self.Model = CRFModel
        elif method == 'bi-lstm-crf':
            pass
        else:
            raise Exception('unknown training method')

    def train(self, train_file, model_file):
        dataset = Dataset(train_file)
        model = self.Model(dataset)
        model.train(model_file)

    def segment(self, model_file, test_file, result_file):
        model = self.Model.load_model(model_file)

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
                    [result_map[tag](word) for word, tag in zip(sentence, model.predict(sentence))]
                ).strip() + '\n')

        # std_result = [sentence.replace('  ', ' ').strip() for sentence in std_result if sentence]

    def evaluate(self, test_file, result_file):
        precision, recall, f1 = eval_func(result_file, test_file)
        print(precision, recall, f1)
