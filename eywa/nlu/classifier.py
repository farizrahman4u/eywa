from ..lang import Document
from ..math import vector_sequence_similarity, euclid_similarity, softmax
from collections import defaultdict
import numpy as np

class Classifier(object):

    def __init__(self):
        self.X = []
        self.Y = []
        self.data = {}
        self.weights = np.array([0.5, .1, .1, 1., .05])
        pass

    def fit(self, X, Y):
        if type(X) in (str, Document):
            X = [X]
            Y = [Y]
        if type(Y) not in (list, tuple):
            Y = [Y] * len(X)
        for x, y in  zip(X, Y):
            x = Document(x)
            # need these for quick eval
            self.X.append(x)
            self.Y.append(y)
            if y in self.data:
                self.data[y].append(x)
            else:
                self.data[y] = [x]

    def predict(self, x, return_probs=False):
        if type(x) in (list, tuple):
            return type(x)([self.predict(i, return_probs) for i in x])
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
        if return_probs:
            scores = softmax(scores)
            return {z[0]: z[1] for z in zip(classes, scores)}
        return classes[np.argmax(scores)]


    def similarity(self, x1, x2):
        #if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0
        score1 = lambda : euclid_similarity(x1.embedding, x2.embedding)
        score2 = lambda : np.dot(x1.embedding, x2.embedding)
        score3 = lambda : vector_sequence_similarity(x1.embeddings, x2.embeddings, self.weights[0], 'dot')
        score4 = lambda : vector_sequence_similarity(x1.embeddings, x2.embeddings, self.weights[0], 'euclid')
        scores = [score1, score2, score3, score4]
        score_weights = self.weights[1:5]
        score = 0.
        for s, w in zip(scores, score_weights):
            if w > 0.05:
                score += s()
        return score * 0.25

    def evaluate(self, X=None, Y=None):
        if X is None:
            X = self.X
            Y = self.Y
        acc = 0.
        err = 0.
        Y_pred = self.predict(X, True)
        for y, y_pred in zip(Y, Y_pred):
            if y == max(y_pred, key=y_pred.__getitem__):
                acc += 1.
            err -= np.log(y_pred[y])
        acc /= len(X)
        return err, acc

    def serialize(self):
        config = {}
        config['X'] = [str(x) for x in self.X]
        config['Y'] = self.Y[:]
        config['weights'] = [float(w) for w in self.weights] 
        return config

    @classmethod
    def deserialize(cls, config):
        clf = cls()
        clf.fit(config['X'], config['Y'])
        clf.weights = np.array(config['weights'])
        return clf
