from ..lang import Document, Token
from ..math import euclid_similarity, vector_sequence_similarity
from ..lang import stop_words
import numpy as np


class Memory(object):

    def __init__(self):
        self.docs = []
        self.weights = np.array([0.5, .1, .1, 1., .0, 0.5, 0.5])
        self.X = []
        self.Y = []
        self._changed = False
        self._index_cache = []
        self._THRESHOLD = 0.4 ## Magic number ?

    def add(self, x):
        if type(x) in (tuple, list):
            self.docs += [Document(i) for i in x]
        else:
            self.docs.append(Document(x))

    def fit(self, X, Y):
        if type(X) not in (list, tuple):
            X = [X]
        if type(Y) not in (list, tuple):
            Y = [Y]
        assert len(X) == len(Y), "Different number of samples in X and Y."
        for x in X:
            self.X.append(Document(x))
        for y in Y:
            self.Y.append(Document(y))
        self._changed = True

    def _compile(self):
        sim = self.similarity
        self._index_cache = []
        app = self._index_cache.append
        rng = list(range(len(self.docs)))
        get_doc = self.docs.__getitem__
        sim = self.similarity
        for y in self.Y:
            app(max(rng, key=lambda i: sim(y, get_doc(i))))

    def clear(self):
        self.docs = []

    def ask(self, q):
        if self._changed:
            self._compile()
            self._changed = False
        q = Document(q)
        docs = self.docs
        sim = self.similarity
        scores = [sim(q, x) for x in docs]
        maxid = np.argmax(scores)
        best_doc = docs[maxid]
        score = scores[maxid]
        ans = self.extract_answer(q, best_doc)
        return ans, best_doc, score


    def similarity(self, x, y):
        # for now we simply count common words
        if not len(x):
            return 0
        if not len(y):
            return 0
        sim = 0.
        for wx in x:
            for wy in y:
                if wx not in stop_words and wy not in stop_words:
                    sim += euclid_similarity(wx.embedding, wy.embedding)
        sim *= (2. / (len(x) + len(y)))
        return sim

    def _extract_by_analogy(self, q, d, x, y):
        q_embs = q.embeddings
        d_embs = d.embeddings
        x_embs = x.embeddings
        y_embs = y.embeddings
        yd = np.dot(y_embs, d_embs.T)  #ny, nd
        xy = np.dot(x_embs, y_embs.T)  #nx, ny
        qx = np.dot(q_embs, x_embs.T)  #nq, nx
        scores1 = qx.dot(xy.dot(yd))
        scores1 = scores1.sum(0)
        scores2 = yd.sum(0)
        gate = self.weights[5]
        scores = gate * scores1 + (1. - gate) * scores2
        maxid = int(np.argmax(scores))
        return d[maxid], scores[maxid]

    def extract_answer(self, q, d):
        if self.X:
            sim = self.similarity
            analogy = self._extract_by_analogy
            answers = []
            scores = []
            gate = self.weights[6]
            for x, y in zip(self.X,self.Y):
                a, s = analogy(q, d, x, y)
                score = gate * s + (1. - gate) + sim(q, x)
                answers.append(a)
                scores.append(score)
            maxid = int(np.argmax(scores))
            return answers[maxid]
        answers = []
        for w in d:
            if w not in q and w not in stop_words:
                answers.append(w)
        if not answers:
            return Token('')
        return min(answers, key=lambda x: x.frequency)

    def serialize(self):
        config = {}
        config['docs'] = [str(d) for d in self.docs]
        config['weights'] = [float(w) for w in self.weights]
        return config

    @classmethod
    def deserialize(cls, config):
        docs = config['docs']
        weights = np.array(config['weights'])
        mem = cls()
        mem.add(docs)
        mem.weights = weights
        return mem

