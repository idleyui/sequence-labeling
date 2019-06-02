#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" HMM CWS Model """

import pickle
from collections import Counter
import numpy as np
from hmmlearn.hmm import MultinomialHMM
from dataset import Dataset


class Model:

    def __init__(self, dataset: Dataset):
        self.word_list = dataset.words
        self.state_list = dataset.states
        self.vocab = dataset.vocab
        self.startprob = None
        self.transmat = None
        self.emissionprob = None
        self.states = ['S', 'B', 'M', 'E']
        self.hmm = MultinomialHMM(n_components=len(self.states))

    """ Use train or load_model to get startprob, transmat and emissionprob """

    def train(self, file):
        self.startprob, self.transmat, self.emissionprob = \
            self._init_state(), self._trans_state(), self._emit_state()
        self.hmm.startprob_ = self.startprob
        self.hmm.transmat_ = self.transmat
        self.hmm.emissionprob_ = self.emissionprob

        if file is not None:
            with open(file, 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def load_model(file: str = None):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def predict_line(self, sentence: str):
        seen_n = np.array([self.preprocess(w) for w in sentence])
        _, b = self.hmm.decode(seen_n, algorithm='viterbi')
        return [self.states[x] for x in b]

    def predict(self, sentence: str):
        """segment"""
        seen_n = np.array([self.preprocess(w) for w in sentence])
        _, b = self.hmm.decode(seen_n, algorithm='viterbi')
        # states = map(lambda x: self.states[x], b)
        states = [self.states[x] for x in b]
        return states

    """Methods calculate startprob, transmat and emissionprob"""

    def _init_state(self):
        """calculate init state"""
        init_counts = {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0}
        for state in self.state_list:
            init_counts[state[0]] += 1.0
        sentence_cnt = len(self.word_list)
        # init_state = {k: log((v+1)/words_count) for k, v in init_counts.items()}
        # plus one smooth
        # init_state = {k: (v + 1) / sentence_cnt for k, v in init_counts.items()}
        init_state = {k: (v + 1) / sentence_cnt for k, v in init_counts.items()}
        return np.array([init_state[s] for s in self.states])

    def _trans_state(self):
        """calculate trans state"""
        trans_counts = {'S': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'B': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'M': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'E': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0}}
        counter = Counter(''.join(self.state_list))
        for line in self.state_list:
            for w1, w2 in zip(line, line[1:]):
                trans_counts[w1][w2] += 1.0
        # trans_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in trans_counts.items()}
        trans_state = {k: {kk: (vv + 1) / counter[k] for kk, vv in v.items()} for k, v in trans_counts.items()}
        return np.array([[trans_state[s][ss] for ss in self.states] for s in self.states])

    def _emit_state(self):
        """calculate emit state"""
        word_dict = {word: 0.0 for word in ''.join(self.vocab)}
        emit_counts = {'S': dict(word_dict), 'B': dict(word_dict), 'M': dict(word_dict), 'E': dict(word_dict)}
        state = ''.join(self.state_list)
        counter = Counter(state)
        for index in range(len(self.state_list)):
            for i in range(len(self.state_list[index])):
                emit_counts[self.state_list[index][i]][self.word_list[index][i]] += 1
        # emit_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in emit_counts.items()}

        emit_state = {k: {kk: (vv + 1) / counter[k] for kk, vv in v.items()} for k, v in emit_counts.items()}

        vocabs = []
        for s in self.states:
            vocabs.extend([k for k, v in emit_state[s].items()])
        vocabs = sorted(list(set(vocabs)))
        self.vocabs = vocabs
        emit_p = np.array([[emit_state[s][w] for w in vocabs] for s in self.states])
        return emit_p

    def preprocess(self, sentence: list):
        """handle oov"""
        return [self.vocabs.index(word) if word in self.vocabs
                else len(self.vocabs) - 1
                for word in sentence]
