import pycrfsuite

from collections import Counter
from dataset import Dataset


class CRF:
    def __init__(self):
        self.tagger = pycrfsuite.Tagger()

    def train(self, X, Y, file):
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X, Y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 200,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(file)

    @staticmethod
    def load_model(file: str):
        model = CRF()
        model.tagger.open(file)
        info = model.tagger.info()

        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(15))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-15:])

        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))

        print("Top positive:")
        print_state_features(Counter(info.state_features).most_common(20))

        print("\nTop negative:")
        print_state_features(Counter(info.state_features).most_common()[-20:])
        return model

    def predict(self, X):
        y_pred = self.tagger.tag(X)
        return y_pred
