from ..math import euclid_similarity, vector_sequence_similarity
from ..lang import todoc
import numpy as np


class Comparator(object):

    def __init__(self):
        self.weights = np.array([0.5, .1, .1, 1., .05])

    def _similarity(self, x1, x2):
        # if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0.
        weights = self.weights
        w0 = weights[0]

        def score1(): return euclid_similarity(x1.embedding, x2.embedding)

        def score2(): return np.dot(x1.embedding, x2.embedding)

        def score3(): return vector_sequence_similarity(x1.embeddings, x2.embeddings, w0, 'dot')

        def score4(): return vector_sequence_similarity(x1.embeddings, x2.embeddings, w0, 'euclid')
        scores = [score1, score2, score3, score4]
        score_weights = weights[1:5]
        score = 0.
        for s, w in zip(scores, score_weights):
            if w > 0.05:
                score += s()
        return score * 0.25

    def __call__(self, x, y):
        x = todoc(x)
        y = todoc(y)
        return self._similarity(x, y)

    def serialize(self):
        config = {}
        config['weights'] = [float(w) for w in self.weights]
        return config

    @classmethod
    def deserialize(cls, config):
        comp = cls()
        comp.weights = np.array(config['weights'])
        return comp
