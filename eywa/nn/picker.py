import tensorflow as tf
from tensorflow import keras
from ..lang import Document
import numpy as np
import math


class NNPicker(object):

    def __init__(self, docs, values, model=None, model_fn=None, training_config=None):
        assert isinstance(docs, (list, tuple))
        assert isinstance(values, (list, tuple))
        for doc in docs:
            assert isinstance(doc, (Document, str))
        assert len(docs) == len(values)
        for i, val in enumerate(values):
            assert isinstance(val, dict)
            for k, v in val.items():
                assert isinstance(k, str)
                assert isinstance(v, (str, list, tuple, type(None)))
        self.docs = [Document(doc) if isinstance(doc, str) else doc for doc in docs]
        self.values = values
        train = model is None and model_fn is None
        self._build_vocab()
        self._build_arrays()    
        self.model_fn = model_fn
        if model:
            self.model = model
        else:
            self._build_model()
        if training_config is None:
            self.training_config = self._default_training_config()
        else:
            training_config2 = self._default_training_config()
            training_config2.update(training_config)
            self.training_config = training_config2
        if train:
            self.fit()

    def _default_training_config(self):
        return {
            'epochs': 100,
            'batch_size': 3
        }

    def _build_vocab(self):
        vocab = set()
        for doc in self.docs:
            for w in doc:
                if not w.is_stop_word:
                    vocab.add(w.text.lower())
        self.vocab = list(vocab)
        self.vocab_map = {w: i for (i, w) in enumerate(self.vocab)}
        self.max_input_length = max(map(len, self.docs))
        keys = set()
        for val in self.values:
            for k in val:
                keys.add(k)
        self.keys = list(keys)
        self.keys_map = {k : i for i, k in enumerate(self.keys)}

    def _get_casing_vector(self, word, vector=None):
        vec_size = 3 # ['NoDigits', 'SomeDigits', 'AllDigits']
        if vector is None:
            vector = np.zeros(vec_size)
        else:
            assert vector.shape == (vec_size, )
        if word.isdigit():
            vector[2] = 1
            return vector
        for c in word:
            if c.isdigit():
                vector[1] = 1
                return vector
        vector[0] = 1
        return vector

    def _get_vocab_vector(self, word, vector=None):
        if not self.vocab:
            return None
        word = word.lower()
        if vector is None:
            vector = np.zeros(len(self.vocab))
        else:
            assert vector.shape == (len(self.vocab),)
        idx = self.vocab_map.get(word)
        if idx:
            vector[idx] = 1
        return vector

    def _vectorize_doc(self, doc, vector=None):
        if not isinstance(doc, Document):
            doc = Document(doc)
        emb_size = len(doc.embedding)
        vec_shape = (len(doc), emb_size + len(self.vocab) + 3)
        if vector is None:
            vector = np.zeros(vec_shape)
        else:
            assert vector.shape == vec_shape
        for i, w in enumerate(doc):
            vector[i][:emb_size] = w.embedding.numpy()
            self._get_vocab_vector(w.text, vector[i][emb_size: emb_size + len(self.vocab)])
            self._get_casing_vector(w.text, vector[i][-3:])
        return vector

    def _vectorize_values(self, doc, values, vector=None):
        vec_shape = (len(doc), len(self.keys) + 1)
        if vector is None:
            vector = np.zeros(vec_shape)
        else:
            assert vector.shape == vec_shape
        vector[:, 0] = 1
        for i, w in enumerate(doc):
            for k, v in values.items():
                k_id = self.keys_map[k]
                if v is None:
                    pass
                elif isinstance(v, str):
                    if v.lower() == w.text.lower():
                        vector[i][k_id + 1] = 1
                        vector[i][0] = 0
                elif isinstance(v, (tuple, list)):
                    for vi in v:
                        if vi.lower() == w.text.lower():
                            vector[i][k_id + 1] = 1
                            vector[i][0] = 0
                            break
        return vector

    # def _class_weight_from_freqs(self, labels_dict, mu=0.15):
    #     total = np.sum(list(labels_dict.values()))
    #     keys = labels_dict.keys()
    #     class_weight = dict()
    #     for key in keys:
    #         score = math.log(mu*total/float(labels_dict[key]))
    #         class_weight[key] = score if score > 1.0 else 1.0
    #     return class_weight

    def _build_arrays(self):
        X = np.zeros((len(self.docs), self.max_input_length, self._vectorize_doc(self.docs[0]).shape[-1]))
        Y = np.zeros((len(self.docs), self.max_input_length, len(self.keys) + 1))
        for i, doc in enumerate(self.docs):
            self._vectorize_doc(doc, X[i][-len(doc):])
        for i, (doc, values) in enumerate(zip(self.docs, self.values)):
            self._vectorize_values(doc, values, Y[i][-len(doc):])
        self.X = X
        self.Y = Y
        class_freqs = Y.sum(axis=(0, 1))
        #print("CLASS FREQS ", class_freqs)
        total = class_freqs.sum()
        mu = 0.15
        class_weight = mu * total / class_freqs
        #class_weight[class_weight < 1.0] = 1.0
        #class_weight[0] *= 0.6
        self.class_weight = class_weight


    def _default_model(self):
        inp = keras.layers.Input((None, self.X.shape[-1]))
        x = keras.layers.Bidirectional(keras.layers.GRU(100, return_sequences=True, recurrent_dropout=0.2))(inp)
        x = keras.layers.Bidirectional(keras.layers.GRU(50, return_sequences=True, recurrent_dropout=0.2))(x)
        x = keras.layers.Activation('relu')(x)
        out = keras.layers.Dense(self.Y.shape[-1], activation='softmax')(x)
        model = keras.models.Model(inp, out)
        model.compile(loss='categorical_crossentropy', optimizer='nadam')
        return model

    def _build_model(self):
        if self.model_fn:
            self.model = self.model_fn(len(self._vectorize_doc(self.docs[0])), len(self.keys) + 1)
        else:
            self.model = self._default_model()

    def fit(self):
        self.model.fit(self.X, self.Y, verbose=0, **self.training_config)

    def predict(self, doc, multiple=False, keys=None, return_scores=False):
        if not isinstance(doc, Document):
            doc = Document(doc)
        inp_vec = self._vectorize_doc(doc)
        out = self.model.predict(np.expand_dims(inp_vec, 0))[0]
        #out *= self.class_weight
        #print(out)
        values = {}
        if multiple:
            for word, pred in zip(doc, out):
                k_id = np.argmax(pred)
                if k_id:
                    score = pred[k_id]
                    new_val = (word.text, score)
                    key = self.keys[k_id - 1]
                    if keys and key not in keys:
                        return
                    if key in values:
                        values[key].append(new_val)
                    else:
                        values[key] = [new_val]
            for k in values:
                values[k].sort(key=lambda x: -x[1])
                if not return_scores:
                    values[k] = [x[0] for x in values[k]]
        else:
            for word, pred in zip(doc, out):
                k_id = np.argmax(pred)
                if k_id:
                    score = pred[k_id]
                    new_val = (word.text, score)
                    key = self.keys[k_id - 1]
                    if keys and key not in keys:
                        return
                    if key in values:
                        prev_val = values[key]
                        if new_val[1] > prev_val[1]:
                            values[key] = new_val
                    else:
                        values[key] = new_val
            if not return_scores:
                for k in values:
                    values[k] = values[k][0]
        return values


    def serialize(self):
        config = {}
        config['docs'] = [str(doc) for doc in self.docs]
        config['values'] = self.values
        config['model'] = self.model.get_config()
        config['weights'] = [w.tolist() for w in self.model.get_weights()]
        config['training_config'] = self.training_config
        return config

    @classmethod
    def deserialize(cls, config):
        model_config = config['model']
        try:
            model = keras.models.Model.from_config(model_config)
        except Exception as e:
            warnings.warn('Model loading failed! ' + e)
            model = None
        if model:
            weights = [np.asarray(w) for w in config['weights']]
            model.set_weights(weights)
        return cls(config['docs'],
                   config['values'],
                   model=model,
                   training_config=config['training_config'])