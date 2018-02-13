from ..lang import Document
from ..math import euclid_similarity, vector_sequence_similarity
import numpy as np

class Memory(object):

    def __init__(self):
        self.docs = []
        self.weights = np.array([0.5, .1, .1, 1., .05])

    def add(self, x):
        if type(x) in (tuple, list):
            self.docs += [Document(i) for i in x]
        else:
            self.docs.append(Document(x))

    def clear(self):
        self.docs = []


    def ask(self, q):
        # token search
        q = Document(q)
        scores = np.zeros(len(self.docs))
        for i, doc in enumerate(self.docs):
            scores[i] = self.token_similarity(q, doc)
        if not scores.sum():
            return None
        idxs = list(range(len(self.docs)))
        idxs.sort(key=scores.__getitem__)
        idxs.reverse()
        ####
        best_doc = self.docs[idxs[0]]
        candidate_tokens = []
        for w in best_doc:
            if w not in q:
                candidate_tokens.append(w)
        return min(candidate_tokens, key=lambda x: x.frequency)

    def token_similarity(self, x1, x2):
        sim = 0.
        freqs = [w.frequency for w in x1] + [w.frequency for w in x2]
        max_freq = np.max(freqs)
        if max_freq:
            for w1 in x1:
                for w2 in x2:
                    if w1.text.lower() == w2.text.lower():
                            freq_inv = 1.- (((w1.frequency + w2.frequency) / (max_freq * 2)))
                            sim += freq_inv
        else:
            for w1 in x1:
                for w2 in x2:
                    if w1.text.lower() == w2.text.lower():
                        sim += 1
        return sim


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

    def ask_yes_no(self, q):
        pass
