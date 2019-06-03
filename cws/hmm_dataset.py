#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from dataset import Dataset


class HMMDataset:
    def train_args(self, file):
        dataset = Dataset(file)
        return dataset.words, dataset.states, dataset.vocab, ['S', 'B', 'M', 'E']

    def test_args(self, sentence):
        return sentence
