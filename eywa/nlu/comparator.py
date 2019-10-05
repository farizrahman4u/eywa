from ..math import euclid_similarity, vector_sequence_similarity
from ..lang import todoc, Document
import tensorflow as tf


class Comparator(object):

    def __init__(self):
        self.weights = [tf.Variable(w, dtype='float32')
                        for w in self.__class__.default_weights()]

    @staticmethod
    def default_weights():
        return [0.5, .1, .1, 1., .05]

    def forward(self, x1, x2):
        assert isinstance(x1, Document)
        assert isinstance(x2, Document)
        # if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0.
        weights = self.weights
        w0 = weights[0]

        def score1():
            return euclid_similarity(x1.embedding, x2.embedding)

        def score2():
            return tf.tensordot(x1.embedding, x2.embedding, 1)

        def score3():
            return vector_sequence_similarity(
                x1.embeddings, x2.embeddings, w0, 'dot')

        def score4():
            return vector_sequence_similarity(
                x1.embeddings, x2.embeddings, w0, 'euclid')
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
        return float(self.forward(x, y))

    def serialize(self):
        config = {}
        config['weights'] = [w.tolist() for w in self.get_weights()]
        return config

    @classmethod
    def deserialize(cls, config):
        comp = cls()
        comp.set_weights(config['weights'])
        return comp

    def set_weights(self, weights):
        assert isinstance(weights, list)
        assert len(weights) == len(self.weights)
        for (w_in, w_curr) in zip(weights, self.weights):
            w_curr.assign(w_in)

    def get_weights(self):
        return [w.numpy() for w in self.weights]
