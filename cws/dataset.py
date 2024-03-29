#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Read corpus to n line of (token list, state list) and make vocabulary list """

import codecs

stop_words = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                 u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")


class Dataset:
    def __init__(self):
        self.words = []
        self.states = []
        self.vocab = []

    def read_corpus(self, file, remove_stop_words=False):
        """read training corpus to word list
        :returns
        words - list
        [
            token1token2token3
            token1token2...
            ...
        ]
        states is same as words
        vocab: [word1, word2, ...]
        """

        with codecs.open(file, encoding='utf8') as f:
            for line in f.readlines():
                word_list = line.strip().split()
                if len(word_list) == 0:
                    continue
                self.words.append(''.join([word for word in word_list]))
                self.states.append(''.join([word2states(word) for word in word_list]))

        self.vocab = sorted(list(set(''.join(self.words)))) + [u'<UNK>']

    def train_args(self, file):
        pass

    def test_args(self, sentence):
        pass

    def decode(self, sentence, states):
        result_map = {
            'S': lambda w: w + ' ',
            'B': lambda w: w,
            'M': lambda w: w,
            'E': lambda w: w + ' '
        }
        return ''.join([result_map[tag](word) for word, tag in zip(sentence, states)]).strip()


def word2states(word):
    if len(word) == 1:
        return 'S'
    else:
        return 'B' + 'M' * (len(word) - 2) + 'E'


def filter(sentence: str) -> str:
    return ''.join([w for w in sentence if w not in stop_words])
