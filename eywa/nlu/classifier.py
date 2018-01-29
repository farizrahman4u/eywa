from ..lang import Document
from ..math import vector_sequence_similarity
from collections import defaultdict
import numpy as np

class Classifier(object):

    def __init__(self):
        self.data = {}
        pass

    def fit(self, X, Y):
        for x, y in  zip(X, Y):
            x = Document(x)
            if y in self.data:
                self.data[y].append(x)
            else:
                self.data[y] = [x]

    def predict(self, x):
        if type(x) in (list, tuple):
            return type(x)(map(self.predict, x))
        if type(x) is not Document:
            x = Document(x)
        classes = self.data.keys()
        scores = [0.] * len(classes)
        for i, k in enumerate(classes):
            for x2 in self.data[k]:
                scores[i] += self.similarity(x, x2)
        return classes[np.argmax(scores)]

    def similarity(self, x1, x2):
        if x1 == x2:
            return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0
        vs1 = np.array([w.embedding for w in x1])
        vs2 = np.array([w.embedding for w in x2])
        return vector_sequence_similarity(vs1, vs2)

