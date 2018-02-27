from ..lang import Document
from ..math import euclid_similarity, vector_sequence_similarity
from ..lang import stop_words
import numpy as np


class Memory(object):

    def __init__(self):
        self.docs = []
        self.weights = np.array([0.5, .1, .1, 1., .0])

    def add(self, x):
        if type(x) in (tuple, list):
            self.docs += [Document(i) for i in x]
        else:
            self.docs.append(Document(x))

    def clear(self):
        self.docs = []

    def ask(self, q):
        q = Document(q)
        docs = self.docs
        sim = self.similarity
        best_doc = max(docs, key=lambda x: sim(q, x))
        ans = self.extract_answer(q, best_doc)
        return ans, best_doc

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
                    sim += np.dot(wx.embedding, wy.embedding)
        sim *= (2. / (len(x) + len(y)))
        return sim

    def extract_answer(self, q, d):
        answers = []
        for w in d:
            if w not in q and w not in stop_words:
                answers.append(w)
        return min(answers, key=lambda x: x.frequency)
        