from ..math import batch_vector_sequence_similarity, euclid_similarity
from ..math import should_pick, get_token_score
from ..lang import Document, Token, tokenize_by_stop_words
import tensorflow as tf
import numpy as np



class EntityExtractor(object):
    """
    
    Gets the required entities from the input.
    
    """

    

    def __init__(self):
        self.X = []
        self.Y = []
        self.keys = {}
        self._changed = False
        self.weights = [tf.Variable(w, dtype='float32') for w in self.__class__.default_weights()]

    @staticmethod
    def default_weights():
        return [0.5, 0.5, 0.5, 0.5]

    @property
    def entities(self):
        return list(self.keys.keys())

    def fit(self, X, Y):
        """
        Trains the model on given data.
        
        # Arguments
        
        X: Input utterance(s). It could be:
            - `str` (or `list` thereof)
            - `Document` instance (or `list` thereof)
        Y: Target values. `dict` mapping from from entity name (`str`)
          to entity value (`str`)(or list thereof).
          The entity names should be same throughout all `dict` 
          elements and number of `dict` elements should be
          same as the number of elements of X.
          
        # Example
        
        Train an `EntityExtractor` to extract entities "intent" and "place" from utterances
        labels "intent" and "place":
        ```python
        x = ['who was the first president of USA', 'which party got elected last time']
        y = [{'intent': 'politics', 'place': 'USA'}, {'intent': 'politics','place': 'here'}]
        ex = EntityExtractor()
        ex.fit(x, y)
        ```
        """
        x_app = self.X.append
        y_app = self.Y.append
        if not isinstance(X, (list, tuple)):
            X = [X]
        if not isinstance(Y, (list, tuple)):
            Y = [Y]
        for x, y in zip(X, Y):
            x = Document(x)
            x_app(x)
            y_app(y)
        self._changed = True

    def _compile(self):
        keys = set()
        for y in self.Y:
            for k in y:
                keys.add(k)
        self.keys = {k: {'picks': [], 'lefts': [], 'rights': [], 'values': [], 'consts': {}, 'types': set()} for k in
                     keys}
        keys = self.keys
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            for k in keys:
                kk = keys[k]
                if k in y:                    
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

    def predict(self, x, keys=None, return_scores=False):
        """
        Extracts entities for  given input utterance(s).
        
        # Arguments
        
        x: Input utterance(s). It could be:
            - `str` (or `list`/`tuple` thereof)
            - `Document` instance (or `list`/`tuple` thereof)
        `return_scores`: `bool`. Default `False`.
        If `True`, Returns a dict (or list thereof if x is a list) mapping entity names
        to a list of tuples, where each tuple consists of a possible entity value (`str`)
        and a confidence score (`float`).
        Else, returns entity name for entity value with highest confidence per utterance.
        `keys`:list of str.  entities to be extracted. 
        If not specified, all entities will be extracted.
        

        # Returns
        
        if `return_scores` is `True`:
            if `x` is a single utterance:
                Returns a `list` of `dict`s of the
                form (entity name, confidence) for each class,
                sorted by decreasing order of confidence.
            if `x` is a `list`/`tuple` of utterances:
                Returns a `list` of results with 1 result per
                utterance. Each result will be a `list` of
                `dict`s of the form (entity name, confidence) for
                each class, sorted by decreasing order of confidence.
        if `return_scores` is `False`:
            if `x` is a single utterance:
                Returns the predicted label as `dict`.
            if `x` is a `list`/`tuple` of utterances:
                Returns the predicted labels for utterances as `list` of
                `dict`s.
        """

        if self._changed:
            self._compile()
            self._changed = False
        if type(x) in (list, tuple):
            return type(x)(map(lambda x:
                               self.predict(x, keys, return_scores), x))
        if keys is None:
            keys = self.keys.keys()
        x = tokenize_by_stop_words(x)
        y_scores = self.forward(x)
        self_keys = self.keys
        if return_scores:
            entity_prob_dist = {}
            for k in keys:
                kk = self_keys[k]
                should_pick, pick_scores, const_scores = y_scores[k]
                should_pick = 1 if should_pick > 0 else 0
                vals = [self.Y[i][k] for i in kk['picks']] + list(kk['consts'].keys())
                scores = (pick_scores.numpy() * should_pick).tolist()
                scores += (const_scores.numpy() * (1 - should_pick)).tolist()
                entity_probs = sorted(list(zip(vals,scores)), key=lambda x: x[1], reverse=True)
                entity_prob_dist[k] = entity_probs
            return entity_prob_dist
        else:
            y = {}         
            for k in keys:
                kk = self_keys[k]
                should_pick, pick_scores, const_scores = y_scores[k]
                if should_pick > 0:
                    vals = [w.text for w in x]
                    y[k] = vals[int(tf.argmax(pick_scores))]
                else:
                    y[k] = list(kk['consts'].keys())[int(tf.argmax(const_scores))]
            return y

    def forward(self, x):
        assert isinstance(x, Document)
        x_embs = x.embeddings
        X = self.X
        self_keys = self.keys
        weights = self.weights
        keys = self.keys.keys()
        y = {}
        for k in keys:
            kk = self_keys[k]
            pick_scores = tf.zeros(len(x))
            consts = kk['consts']
            const_scores = tf.zeros(len(consts))
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
                pick = tf.zeros(1)
            else:
                if not consts:
                    pick = tf.ones(1)
                else:
                    pick_embs = [X[i].embeddings for i in picks]
                    non_pick_embs = [X[i].embeddings for i in non_picks]
                    vals = [self.Y[i][k] for i in pick_idxs]
                    n = len(vals)
                    c = len(set(vals))
                    variance = float(c) / n
                    pick = should_pick(x_embs, pick_embs, non_pick_embs, variance, weights)
            if pick > 0:
                pick_scores = []
                lefts_embs = [d.embeddings for d in kk['lefts']]
                rights_embs = [d.embeddings for d in kk['rights']]
                vals_embs = [v.embedding for v in kk['values']]
                for i, t in enumerate(x):
                    left = x[:i]
                    right = x[i:]
                    token_score = get_token_score(t.embedding, left.embeddings, right.embeddings,
                                                  lefts_embs, rights_embs, vals_embs, bool(entity_type), weights)
                    pick_scores.append(token_score)
                pick_scores = tf.stack(pick_scores)
            else:
                const_scores = []
                for ck in consts:
                    docs = [X[i] for i in consts[ck]]
                    embs = [doc.embeddings for doc in docs]
                    score = tf.reduce_max(batch_vector_sequence_similarity(embs, x_embs))
                    const_scores.append(score)
                const_scores = tf.stack(const_scores)
            y[k] = pick, pick_scores, const_scores
        return y

    def evaluate(self, X=None, Y=None):
        """Evaluates the EntityExtractor on given data.

        Either both `X` and `Y` arguments should be provided
        or both of them should be left unspecified (`None`).
        If left unspecified, the cumilative data used to train
        the `EntityExtractor` will be used for evaluation .

        # Arguments
        X: Input utterance(s). It could be:
            - `str` (or `list` thereof)
            - `Document` instance (or `list` thereof)
        Y: Target values. `dict` mapping from from entity name (`str`)
          to entity value (`str`)(or list thereof).
          The entity names should be same throughout all `dict` 
          elements and number of `dict` elements should be
          same as the number of elements of X.

        # Returns
        `tuple` of error(`float`) and accuracy(`float`)
        """

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
        """Serializes the `EntityExtractor` object to a json
        friendly config.

        # Returns
        `dict`
        """
        config = {}
        config['X'] = [str(x) for x in self.X]
        config['Y'] = self.Y[:]
        config['weights'] = [w.tolist() for w in self.get_weights()]
        return config

    @classmethod
    def deserialize(cls, config):
        """Deserializes a `EntityExtractor` config to a `EntityExtractor` instance.

        # Arguments
        config: `dict`. `EntityExtractor` config (generated by `EntityExtractor.serialize`).

        # Returns
        `EntityExtractor` instance
        """
        ee = cls()
        ee.fit(config['X'], config['Y'])
        ee.set_weights(config['weights'])
        return ee

    def set_weights(self, weights):
        """Sets weights of the `EntityExtractor` to given
        values.

        # Arguments
        weights: `list` of numpy arrays
        """
        assert isinstance(weights, list)
        assert len(weights) == len(self.weights)
        for (w_in, w_curr) in zip(weights, self.weights):
                w_curr.assign(w_in)

    def get_weights(self):
        """Returns weights of the `EntityExtractor`.

        # Returns
        `list` of numpy arrays
        """
        return [w.numpy() for w in self.weights]
