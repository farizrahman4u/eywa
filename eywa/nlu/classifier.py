from ..lang import Document
from ..math import vector_sequence_similarity, euclid_similarity
from collections import defaultdict
import numpy as np

class Classifier(object):

    def __init__(self):
        self.data = {}
        self.weights = np.array([0.5])
        pass

    def fit(self, X, Y):
        if type(X) in (str, Document):
            X = [X]
            Y = [Y]
        if type(Y) not in (list, tuple):
            Y = [Y] * len(X)
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
        scores = np.zeros(len(classes))
        for i, k in enumerate(classes):
            for x2 in self.data[k]:
                score = self.similarity(x, x2)
                if score > scores[i]:
                    scores[i] = score
        #scores /= np.array([len(self.data[c]) for c in classes])
        return classes[np.argmax(scores)]

    def _similarity(self, x1, x2):
        return euclid_similarity(x1.embedding, x2.embedding)

    def similarity(self, x1, x2):
        #if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0
        return vector_sequence_similarity(x1.embeddings, x2.embeddings, self.weights[0])

