from ..lang import Document
from collections import defaultdict
import numpy as np


class Classifier(object):
    def __init__(self, weights=[]):
        self.data = {}
        self.weights = weights
        self._changed = False

    def fit(self, x, y):
        if type(x) not in (tuple, list):
            x = [x]
        if type(y) not in (tuple, list):
            y = [y]
        assert len(x) == len(y)
        data = self.data
        for s, l in zip(x, y):
            doc = Document(s)
            if l in data:
                data[l].append(doc)
            else:
                data[l] = [doc]
        self._changed = True

    def compare(self, doc1, doc2):
        if doc1.text == doc2.text:
            return 1
        embs1 = [w.embedding for w in doc1]
        embs2 = [w.embedding for w in doc2]
        iv1 = [w.in_vocab for w in doc1]
        iv2 = [w.in_vocab for w in doc2]


    def _compile(self):
        ### bag of words
        buckets = {}
        data = self.data
        vocab = set()
        for k in data:
            counts = defaultdict(lambda: 0)
            buckets[k] = counts
            v = data[k]
            for doc in v:
                for w in doc:
                    vocab.add(w)
                    if w in counts:
                        counts[w] += 1
                    else:
                        counts[w] = 1
        freqs = {}
        for k in buckets:
            counts = buckets[k]
            f = [counts[w] for w in vocab]
            f = np.array(f, dtype=float)
            f -= f.mean()
            std = f.std()
            if std:
                f /= std
            freqs[k] = f
        self._freqs = freqs
        self._vocab = vocab
        ######

    def predict(self, x):
        if self._changed:
            self._compile()
            self._changed = False
        pass

