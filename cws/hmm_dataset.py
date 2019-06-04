#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from dataset import Dataset


class CWSHMMDataset(Dataset):

    def __init__(self):
        super().__init__()

    def train_args(self, file):
        self.read_corpus(file)
        return self.words, self.states, self.vocab, ['S', 'B', 'M', 'E']

    def test_args(self, sentence):
        return sentence
