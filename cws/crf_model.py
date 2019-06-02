from itertools import chain

import pycrfsuite
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from dataset import Dataset


class CRF:
    def __init__(self, dataset: Dataset = None):
        if dataset is not None:
            self.sent = []
            for word_line, state_line in zip(dataset.words, dataset.states):
                self.sent.append([(w[0], w[1]) for i, w in enumerate(zip(word_line, state_line))])
            self.X, self.Y = self.mk_data(self.sent)
        self.tagger = pycrfsuite.Tagger()

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
            [-2, -1],
            [-2, -1, 0],
            [1],
            [0, 1],
            [2],
            [1, 2],
            [0, 1, 2],
            [0]
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

    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )

    @staticmethod
    def load_model(file: str):
        model = CRF()
        model.tagger.open(file)
        return model

    def predict(self, sentence):

        x = self.line2feature(sentence)
        y_pred = self.tagger.tag(x)

        return y_pred
        # one_line = ''
        # for j, w in enumerate(sentence):
        #     if y_pred[j] == 'S':
        #         one_line += (w[0] + ' ')
        #     elif y_pred[j] == 'B':
        #         one_line += w[0]
        #     elif y_pred[j] == 'M':
        #         one_line += w[0]
        #     else:
        #         one_line += (w[0] + ' ')
        # return one_line
        # print(self.bio_classification_report(Y_test, y_pred))

    def _predict(self, sentence):

        # X_test, Y_test = self.mk_data(sent)
        X_test = [self.line2feature(line) for line in [sentence]]
        y_pred = [self.tagger.tag(x) for x in X_test]
        # return y_pred

        lines = []
        for i, line in enumerate([sentence]):
            one_line = ''
            for j, w in enumerate(line):
                if y_pred[i][j] == 'S':
                    one_line += (w[0] + ' ')
                elif y_pred[i][j] == 'B':
                    one_line += w[0]
                elif y_pred[i][j] == 'M':
                    one_line += w[0]
                else:
                    one_line += (w[0] + ' ')
            lines.append(one_line)
        return '\n'.join(lines)

        # print(self.bio_classification_report(Y_test, y_pred))
