import pycrfsuite

from dataset import Dataset


class CWSCRFDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tagger = pycrfsuite.Tagger()

    def train_args(self, file):
        self.read_corpus(file)
        self.sent = []
        for word_line, state_line in zip(self.words, self.states):
            self.sent.append([(w[0], w[1]) for i, w in enumerate(zip(word_line, state_line))])
        self.X, self.Y = self._mk_data(self.sent)
        return self.X, self.Y

    def test_args(self, sentence):
        return self._line2feature(sentence)

    def _word2feature(self, line, i):
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

    def _line2feature(self, line):
        return [self._word2feature(line, i) for i in range(len(line))]

    def _line2labels(sent, line):
        return [line[i][1] for i in range(len(line))]

    def _mk_data(self, sent):
        return [self._line2feature(line) for line in sent], [self._line2labels(line) for line in sent]
