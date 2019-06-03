import pycrfsuite

from collections import Counter
from dataset import Dataset


class CRFDataset:
    def __init__(self, dataset: Dataset = None):
        if dataset is not None:
            self.sent = []
            for word_line, state_line in zip(dataset.words, dataset.states):
                self.sent.append([(w[0], w[1]) for i, w in enumerate(zip(word_line, state_line))])
            self.X, self.Y = self.mk_data(self.sent)
        self.tagger = pycrfsuite.Tagger()

    def train_args(self, file):
        dataset = Dataset(file)
        self.sent = []
        for word_line, state_line in zip(dataset.words, dataset.states):
            self.sent.append([(w[0], w[1]) for i, w in enumerate(zip(word_line, state_line))])
        self.X, self.Y = self.mk_data(self.sent)
        return self.X, self.Y

    def test_args(self, sentence):
        return self.line2feature(sentence)

    def word2feature(self, line, i):
        def check(index):
            return 0 < index < len(line)

        def char_feature(name, offsets, col):
            name = '/'.join([str(o) for o in offsets]) + ':' + name + '='
            values = [line[i + off][col] if check(i + off) else '' for off in offsets]
            return name + '/'.join(values)

        offset_list = [
            [-1],
            [-1, 0],
            [-2],
            # [-2, -1],
            [-2, -1, 0],
            # [1],
            # [0, 1],
            # [2],
            # [1, 2],
            # [0, 1, 2],
            # [0]
        ]

        features = [char_feature('char', offsets, 0) for offsets in offset_list]
        # features.extend([char_feature('tag', offsets, 1) for offsets in offset_list])

        return features

    def line2feature(self, line):
        return [self.word2feature(line, i) for i in range(len(line))]

    def line2labels(sent, line):
        return [line[i][1] for i in range(len(line))]

    def mk_data(self, sent):
        return [self.line2feature(line) for line in sent], [self.line2labels(line) for line in sent]

    def train(self, file):
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(self.X, self.Y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 200,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(file)

