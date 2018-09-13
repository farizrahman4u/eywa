from ..math import batch_vector_sequence_similarity, euclid_similarity
from ..lang import Document, Token
import numpy as np


np_max = np.max
np_argmax = np.argmax

class EntityExtractor(object):

    def __init__(self):
        self.X = []
        self.Y = []
        self.keys = {}
        self._changed = False
        self.weights = np.array([0.5, 0.5, 0.5, 0.5])

    @property
    def entities(self):
        return list(self.keys.keys())

    def fit(self, X, Y):
        x_app = self.X.append
        y_app = self.Y.append
        for x, y in zip(X, Y):
            x = Document(x)
            x_app(x)
            y_app(y)
        self._changed = True

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
                    types_add = types.add
                    indices_app = indices.append
                    for j, t in enumerate(x):
                        if t.text == v:
                            indices_app(j)
                            types_add(t.type)
                    if indices:
                        kk['picks'].append(i)
                        lefts_app = kk['lefts'].append
                        rights_app = kk['rights'].append
                        values_app = kk['values'].append
                        for ind in indices:
                            left = x[:ind]
                            right = x[ind:]
                            lefts_app(left)
                            rights_app(right)
                            values_app(Token(v))
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
        if type(x) in (list, tuple):
            return type(x)(map(self.predict, x))
        if self._changed:
            self.compile()
            self._changed = False
        if keys is None:
            keys = self.keys.keys()
        x = Document(x)
        y = {}
        x_embs = x.embeddings
        X = self.X
        self_keys = self.keys
        for k in keys:
            kk = self_keys[k]
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
            else:
                pick_embs = [X[i].embeddings for i in picks]
                non_pick_embs = [X[i].embeddings for i in non_picks]
                pick_score = np_max(batch_vector_sequence_similarity(pick_embs, x_embs))
                if non_pick_embs:
                    non_pick_score = np_max(batch_vector_sequence_similarity(non_pick_embs, x_embs))
                else:
                    non_pick_score = 0.
                vals = [self.Y[i][k] for i in pick_idxs]
                n = len(vals)
                c = len(set(vals))
                variance = float(c) / n
                pick_bias = self.weights[3]
                s = pick_score + non_pick_score
                pick_score /= s
                non_pick_score /= s
                pick_score *= pick_bias
                pick_score *= variance
                non_pick_score *= 1. - pick_bias
                non_pick_score += 1. - variance
                pick = pick_score >= non_pick_score
            if pick:
                token_scores = []
                for i, t in enumerate(x):
                    lefts_embs = [d.embeddings for d in kk['lefts']]
                    rights_embs = [d.embeddings for d in kk['rights']]
                    left = x[:i]
                    right = x[i:]
                    left_score = np_max(batch_vector_sequence_similarity(lefts_embs, left.embeddings))
                    right_score = np_max(batch_vector_sequence_similarity(rights_embs, right.embeddings))
                    value_score = np_max([euclid_similarity(v.embedding, t.embedding) for v in kk['values']])
                    #value_score = np.mean(np.dot([v.embedding for v in kk['values']], t.embedding))
                    left_right_weight = self.weights[0]
                    word_neighbor_weight = self.weights[1]
                    neighbor_score = left_right_weight * left_score + (1. - left_right_weight) * right_score
                    token_score = word_neighbor_weight * value_score + (1. - word_neighbor_weight) * neighbor_score
                    if entity_type:
                        entity_type_weight = self.weights[2]
                        token_score *= 1. + entity_type_weight
                    token_scores.append(token_score)
                y[k] = x[int(np_argmax(token_scores))].text
            else:
                consts = kk['consts']
                if consts:
                    consts_keys = consts.keys()
                    scores = []
                    for ck in consts:
                        docs = [X[i] for i in consts[ck]]
                        embs = [doc.embeddings for doc in docs]
                        score = np_max(batch_vector_sequence_similarity(embs, x_embs))
                        scores.append(score)
                    y[k] = consts_keys[int(np_argmax(scores))]
                else:
                    docs = [X[i] for i in pick_idxs]
                    embs = [doc.embeddings for doc in docs]
                    best_val_id = np_argmax(batch_vector_sequence_similarity(embs, x_embs))
                    y[k] = vals[best_val_id]
        return y
    
    def serialize(self):
        config = {}
        config['X'] = [str(x) for x in self.X]
        config['Y'] = self.Y[:]
        config['weights'] = [float(w) for w in self.weights] 
        return config

    @classmethod
    def deserialize(cls, config):
        ee = cls()
        ee.fit(config['X'], config['Y'])
        ee.weights = np.array(config['weights'])
        return ee
