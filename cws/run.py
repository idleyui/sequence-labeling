#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""running script for cws """

import argparse
import codecs

from hmm_model import Model as HMMModel
from crf_model import CRF as CRFModel
from dataset import Dataset
from evaluate import eval_func

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='hmm', type=str, help='algorithm: hmm or crf')
parser.add_argument('--mode', default='train', type=str, help='mode: train or eval')
parser.add_argument('--train_file', type=str, help='train file path')
parser.add_argument('--test_file', type=str, help='test file path')
parser.add_argument('--result_file', type=str, help='file path to save result')
parser.add_argument('--model_file', type=str, help='file path to save model')


class CWS:
    def __init__(self, method, train_file, test_file, result_file, model_file):
        if method == 'hmm':
            self.Model = HMMModel
        elif method == 'crf':
            self.Model = CRFModel
        elif method == 'bi-lstm-crf':
            pass
        else:
            raise Exception('unknown training method')

        self.train_file = train_file
        self.test_file = test_file
        self.result_file = result_file
        self.model_file = model_file

    def train(self):
        self._check(self.train_file)
        self._check(self.model_file)
        dataset = Dataset(self.train_file)
        model = self.Model(dataset)
        model.train(self.model_file)

    def segment(self):
        self._check(self.test_file)
        self._check(self.model_file)
        self._check(self.result_file)
        model = self.Model.load_model(self.model_file)

        with codecs.open(self.test_file, encoding='utf8') as f:
            std_result = f.readlines()
            raw_lines = [''.join(line.split()) for line in std_result]

        result_map = {
            'S': lambda w: w + ' ',
            'B': lambda w: w,
            'M': lambda w: w,
            'E': lambda w: w + ' '
        }

        with codecs.open(self.result_file, 'w', encoding='utf8') as f:
            for sentence in raw_lines:
                f.write(''.join(
                    [result_map[tag](word) for word, tag in zip(sentence, model.predict(sentence))]
                ).strip() + '\n')

        # std_result = [sentence.replace('  ', ' ').strip() for sentence in std_result if sentence]

    def evaluate(self):
        self._check(self.test_file)
        self._check(self.result_file)
        precision, recall, f1 = eval_func(self.result_file, self.test_file)
        print(precision, recall, f1)

    def _check(self, obj):
        if None == obj:
            raise Exception('None file')


if __name__ == '__main__':
    args = parser.parse_args()
    # args.train_file = '../data/trainset/train_cws.txt'
    # args.model_file = '../data/result/cws_hmm_model.pkl'
    cws = CWS(args.method, args.train_file, args.test_file, args.result_file, args.model_file)
    if args.mode == 'train':
        cws.train()
    elif args.mode == 'segment':
        cws.segment()
    elif args.mode == 'eval':
        cws.evaluate()
    else:
        print('wrong mode arg, use one of train/segment/eval')
