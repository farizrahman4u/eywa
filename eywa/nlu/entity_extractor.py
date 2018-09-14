from ..math import batch_vector_sequence_similarity, euclid_similarity
from ..math import should_pick, get_token_score
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
        self.keys = {k: {'picks': [], 'lefts': [], 'rights': [], 'values': [], 'consts': {}, 'types': set()}
                     for k in keys}
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
        weights = self.weights
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
                vals = [self.Y[i][k] for i in pick_idxs]
                n = len(vals)
                c = len(set(vals))
                variance = float(c) / n
                pick = should_pick(x_embs, pick_embs, non_pick_embs, variance, weights)
            if pick:
                token_scores = []
                lefts_embs = [d.embeddings for d in kk['lefts']]
                rights_embs = [d.embeddings for d in kk['rights']]
                vals_embs = [v.embedding for v in kk['values']]
                for i, t in enumerate(x):
                    left = x[:i]
                    right = x[i:]
                    token_score = get_token_score(t.embedding, left.embeddings, right.embeddings,
                                                  lefts_embs, rights_embs, vals_embs, bool(entity_type), weights)
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


    def evaluate(self, X=None, Y=None):
        if X is None:
            X = self.X
            Y = self.Y
        errors = 0
        n = 0
        for x, y in zip(X, Y):
            y_pred = self.predict(x)
            for k in y:
                n += 1
                if y[k] != y_pred[k]:
                    errors += 1
        accuracy = 1. - float(errors) / n
        return errors, accuracy


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
