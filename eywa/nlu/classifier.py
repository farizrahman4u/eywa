from ..lang import Document
from ..math import vector_sequence_similarity, euclid_similarity, softmax
from ..blameflow import Node, Switch
from ..blameflow import Blame, BlameType
from collections import defaultdict
import tensorflow as tf
import numpy as np


class Classifier(Switch):

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.X = []
        self.Y = []
        self.data = {}
        self.weights = [tf.Variable(w, dtype='float32')
                        for w in self.__class__.default_weights()]
        self.grads = {}
        pass

    @staticmethod
    def default_weights():
        return [0.5, .1, .1, 1., .05]

    @property
    def classes(self):
        return list(self.data.keys())

    def fit(self, X, Y):
        if type(X) in (str, Document):
            X = [X]
            Y = [Y]
        if type(Y) not in (list, tuple):
            Y = [Y] * len(X)
        assert len(X) == len(Y), "Different number of samples in X and Y."
        X_app = self.X.append
        Y_app = self.Y.append
        data = self.data
        for x, y in zip(X, Y):
            x = Document(x)
            # need these for quick eval
            X_app(x)
            Y_app(y)
            if y in data:
                data[y].append(x)
            else:
                data[y] = [x]
        self.options = set(self.Y)
        self._changed = True

    def forward(self, x):
        assert isinstance(x, Document)
        scores = []
        f = self._similarity
        for k, v in self.data.items():
            with tf.GradientTape() as tape:
                score = f(x, v[0])
                for x2 in v[1:]:
                    score = max(score, f(x, x2))
            self.grads[k] = tape.gradient(score, self.weights)
            scores.append(score)
        scores = tf.stack(scores)
        return scores

    def predict(self, x, return_scores=False):
        if type(x) in (list, tuple):
            return type(x)([self.predict(i, return_scores) for i in x])
        if not isinstance(x, Document):
            x = Document(x)
        classes = list(self.data.keys())
        scores = self.forward(x)
        if return_scores:
            probs_dist = sorted({z[0]: float(z[1]) for z in zip(
                classes, scores)}.items(), key=lambda x: x[1], reverse=True)
            return probs_dist
        return classes[np.argmax(scores.numpy())]

    def _similarity(self, x1, x2):
        # if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0.
        weights = self.weights
        w0 = weights[0]

        def score1():
            return euclid_similarity(x1.embedding, x2.embedding)

        def score2():
            return tf.tensordot(x1.embedding, x2.embedding, (0, 0))

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
        config = super(Classifier, self).serialize()
        config['X'] = [str(x) for x in self.X]
        config['Y'] = self.Y[:]
        config['weights'] = [w.tolist() for w in self.get_weights()]
        return config

    @classmethod
    def deserialize(cls, config):
        weights = config.pop('weights')
        X, Y = config.pop('X'), config.pop('Y')
        clf = super(Classifier, cls).deserialize(config)
        clf.set_weights(weights)
        clf.fit(X, Y)
        return clf

    def set_weights(self, weights):
        assert isinstance(weights, list)
        assert len(weights) == len(self.weights)
        for (w_in, w_curr) in zip(weights, self.weights):
            w_curr.assign(w_in)

    def get_weights(self):
        return [w.numpy() for w in self.weights]

    def switch_f(self, inputs):
        assert len(inputs) == 1, "Classifier is a single input node type."
        inp = list(inputs.values())[0]
        return self.predict(inp)

    def blame(self, blame):
        super(Classifier, self).blame(blame)
        if blame.blame_type == BlameType.POSITIVE:
            for g, w in zip(self.grads[self.value], self.weights):
                w.assign_add(g * 0.05)
            blame.node_updated = True
        elif blame.blame_type == BlameType.NEGATIVE:
            for g, w in zip(self.grads[self.value], self.weights):
                w.assign_sub(g * 0.05)
