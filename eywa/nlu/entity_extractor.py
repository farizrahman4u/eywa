from ..math import batch_vector_sequence_similarity, euclid_similarity
from ..lang import Document, Token
import numpy as np


class EntityExtractor(object):

    def __init__(self):
        self.X = []
        self.Y = []
        # info about entities
        # coloumns:
        # #entity #pick? #entity_type_counts #values
        self.keys = {}
        self._changed = False
        self.weights = np.array([0.5, 0.5, 0.5])
        pass

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            x = Document(x)
            self.X.append(x)
            self.Y.append(y)
        self._changed = True

    def similarity(self, x1, x2):
        if x1 == x2:
            return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0
        vs1 = np.array([w.embedding for w in x1])
        vs2 = np.array([w.embedding for w in x2])
        return vector_sequence_similarity(vs1, vs2)

    def compile(self):
        # create a profile for each 'key'
        keys = set()
        for y in self.Y:
            for k in y:
                keys.add(k)
        self.keys = {k: {'picks': [], 'lefts': [], 'rights': [], 'values': [], 'consts': {}, 'types': set()} for k in keys}
        keys = self.keys
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            for k in keys:
                if k in y:
                    kk = keys[k]
                    types = kk['types']
                    v = y[k]
                    indices = []
                    for j, t in enumerate(x):
                        if t.text == v:
                            indices.append(j)
                            types.add(t.type)
                    if indices:
                        kk['picks'].append(i)
                        for ind in indices:
                            left = x[:ind]
                            right = x[ind:]
                            kk['lefts'].append(left)
                            kk['rights'].append(right)
                            kk['values'].append(Token(v))
                    else:
                        consts = kk['consts']
                        if v in consts:
                            consts[v].append(i)
                        else:
                            consts[v] = [i]
                else:
                    consts = kk['consts']
                    if None in consts:
                        consts[None].append(i)
                    else:
                        consts[None] = [i]

    def predict(self, x, keys=None):
        if self._changed:
            self.compile()
            self._changed = False
        if keys is None:
            keys = self.keys.keys()
        x = Document(x)
        y = {}
        x_embs = x.embeddings
        X = self.X
        for k in keys:
            kk = self.keys[k]
            types = kk['types']
            if len(types) == 1:
                entity_type = list(types)[0]
            else:
                entity_type = None

            pick_idxs = kk['picks']
            picks = []
            non_picks = []
            for i in range(len(X)):
                if i in pick_idxs:
                    picks.append(i)
                else:
                    non_picks.append(i)
            if not picks:
                pick = False
            elif not non_picks:
                pick = True
            else:
                pick_embs = [X[i].embeddings for i in picks]
                non_pick_embs = [X[i].embeddings for i in non_picks]
                pick_score = np.max(batch_vector_sequence_similarity(pick_embs, x_embs))
                non_pick_score = np.max(batch_vector_sequence_similarity(non_pick_embs, x_embs))
                pick = pick_score >= non_pick_score
            if pick:
                token_scores = []
                for i, t in enumerate(x):
                    lefts_embs = [d.embeddings for d in kk['lefts']]
                    rights_embs = [d.embeddings for d in kk['rights']]
                    left = x[:i]
                    right = x[i:]
                    left_score = np.max(batch_vector_sequence_similarity(lefts_embs, left.embeddings))
                    right_score = np.max(batch_vector_sequence_similarity(rights_embs, right.embeddings))
                    value_score = np.max([euclid_similarity(v.embedding, t.embedding) for v in kk['values']])
                    #value_score = np.mean(np.dot([v.embedding for v in kk['values']], t.embedding))
                    left_right_weight = self.weights[0]
                    word_neighbor_weight = self.weights[1]
                    neighbor_score = left_right_weight * left_score + (1. - left_right_weight) * right_score
                    token_score = word_neighbor_weight * value_score + (1. - word_neighbor_weight) * neighbor_score
                    if entity_type:
                        entity_type_weight = self.weights[2]
                        token_score *= 1. + entity_type_weight
                    token_scores.append(token_score)
                y[k] = x[int(np.argmax(token_scores))].text
            else:
                consts = kk['consts']
                consts_keys = consts.keys()
                scores = []
                for ck in consts:
                    docs = [C[i] for i in consts[ck]]
                    embs = [doc.embeddings for doc in docs]
                    score = np.max(batch_vector_sequence_similarity(embs, x_embs))
                    scores.append(score)
                y[k] = consts_keys[np.argmax(scores)]
        return y
            

                    


        
