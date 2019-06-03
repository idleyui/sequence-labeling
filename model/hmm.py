#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" HMM Model """

import pickle
from collections import Counter
import numpy as np
# todo replace this
from hmmlearn.hmm import MultinomialHMM


class HMM:

    def __init__(self):
        pass

    def train(self, obs_seq_list: list, state_seq_list: list, obs_set: list, state_set: list, file):
        """
        :param obs_seq_list: observation sequence list [[o1, o2, o3], [o1, o2, o3]...]
        :param state_seq_list: state sequence list [[s1, s2, s3], [s1, s2, s3]...]
        :param obs_set: all possible observation state
        :param state_set: all possible state
        """
        self.obs_seq_list = obs_seq_list
        self.state_seq_list = state_seq_list
        self.obs_set = obs_set
        self.state_set = state_set
        self.counter = Counter(''.join(state_seq_list))

        self.hmm = MultinomialHMM(n_components=len(self.state_set))

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

    def predict(self, obs):
        obs_seq = np.array([self.preprocess(o) for o in obs])
        _, b = self.hmm.decode(obs_seq, algorithm='viterbi')
        states = [self.state_set[x] for x in b]
        return states

    """Methods calculate startprob, transmat and emissionprob"""

    def _init_state(self):
        """calculate init state"""
        first_states = [s[0] for s in self.state_seq_list]
        cnt = Counter(first_states)
        seq_amount = len(first_states)
        # init_state = {k: log((v+1)/words_count) for k, v in init_counts.items()}
        # plus one smooth
        init_state = [(cnt[s] + 1) / seq_amount for s in self.state_set]
        return np.array(init_state)

    def _trans_state(self):
        """calculate trans state"""
        end_state_cnt = {state: 0 for state in self.state_set}
        # trans_cnt[start_state][end_state]
        trans_cnt = {state: dict(end_state_cnt) for state in self.state_set}
        for line in self.state_seq_list:
            for w1, w2 in zip(line, line[1:]):
                trans_cnt[w1][w2] += 1.0
        # trans_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in trans_counts.items()}
        trans_matrix = [
            [(trans_cnt[start_s][end_s] + 1) / self.counter[start_s]
             for end_s in self.state_set]
            for start_s in self.state_set
        ]
        return np.array(trans_matrix)

    def _emit_state(self):
        """calculate emit state"""
        obs_dict = {word: 0.0 for word in self.obs_set}
        emit_cnt = {state: dict(obs_dict) for state in self.state_set}
        for state_seq, obs_seq in zip(self.state_seq_list, self.obs_seq_list):
            for state, obs in zip(state_seq, obs_seq):
                emit_cnt[state][obs] += 1
        # emit_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in emit_counts.items()}

        emit_matrix = [[(emit_cnt[s][o] + 1) / self.counter[s] for o in self.obs_set] for s in self.state_set]
        return np.array(emit_matrix)

    def preprocess(self, seq: list):
        """handle new observation"""
        return [self.obs_set.index(obs) if obs in self.obs_set
                else len(self.obs_set) - 1
                for obs in seq]
