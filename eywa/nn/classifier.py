import tensorflow as tf
from tensorflow import keras
from ..lang import Document
import numpy as np
import warnings


class NNClassifier(object):
    def __init__(self, docs, labels, model_fn=None, model=None, training_config=None):
        assert isinstance(docs, (list, tuple))
        assert isinstance(labels, (list, tuple))
        for doc in docs:
            assert isinstance(doc, (Document, str))
        for label in labels:
            assert isinstance(label, str)
        assert len(docs) == len(labels)
        self.docs = [Document(doc) if isinstance(doc, str) else doc for doc in docs]
        self.labels = labels
        self.classes = list(set(labels))
        self.classes_map = {c : i for i, c in enumerate(self.classes)}
        assert len(self.classes) >= 2
        self.is_binary = len(self.classes) == 2
        train = model is None and model_fn is None
        self.model_fn = model_fn
        self._build_vocab()
        self._build_arrays()
        if model is None:
            self._build_model()
        else:
            self.model = model
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
            'epochs': 20,
            'batch_size': 3
        }

    def _vectorize(self, doc, vector=None):
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

    def _build_vocab(self):
        vocab = set()
        for doc in self.docs:
            for w in doc:
                if not w.is_stop_word:
                    vocab.add(w.text.lower())
        self.vocab = list(vocab)
        self.vocab_map = {w: i for (i, w) in enumerate(self.vocab)}
        self.max_input_length = max(map(len, self.docs))

    def _build_arrays(self):
        X = np.zeros((len(self.docs), self.max_input_length, self._vectorize(self.docs[0]).shape[1]))
        for i, doc in enumerate(self.docs):
            self._vectorize(doc, X[i][-len(doc):])
        if self.is_binary:
            Y = np.zeros(len(self.docs))
            for i, label in enumerate(self.labels):
                class_id = self.classes_map[label]
                Y[i] = class_id
        else:
            Y = np.zeros((len(self.docs), len(self.classes)))
            for i, label in enumerate(self.labels):
                class_id = self.classes_map[label]
                Y[i][class_id] = 1
        self.X = X
        self.Y = Y

    def _default_model(self):
        inp = keras.layers.Input((None, self.X.shape[-1]))
        x = keras.layers.GRU(100, return_sequences=True, recurrent_dropout=0.2)(inp)
        x = keras.layers.GRU(50, recurrent_dropout=0.2)(x)
        if self.is_binary:
            out = keras.layers.Dense(1, activation='sigmoid')(x)
        else:
            out = keras.layers.Dense(len(self.classes), activation='softmax')(x)
        model = keras.models.Model(inp, out)
        if self.is_binary:
            model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        else:
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model

    def _build_model(self):
        if self.model_fn is not None:
            self.model = self.model_fn(self.X.shape[-1], len(self.classes))
        else:
            self.model = self._default_model()


    def fit(self):
        self.model.fit(self.X, self.Y, **self.training_config)

    def predict(self, doc):
        inp_vec = self._vectorize(doc)
        out = self.model.predict(np.expand_dims(inp_vec, 0))[0]
        if self.is_binary:
            return self.classes[1] if out > 0.5 else self.classes[0]
        else:
            return self.classes[np.argmax(out)]

    def predict_prob(self, doc):
        if not isinstance(doc, Document):
            doc = Document(doc)
        inp_vec = self._vectorize(doc)
        out = self.model.predict(np.expand_dims(inp_vec, 0))[0]
        
        if self.is_binary:
            prob = {
                    self.classes[1] : out,
                    self.classes[0] : 1. - out
            }
        else:
            prob = {
                c: out[i] for i, c in enumerate(self.classes)
            }
        return prob

    def serialize(self):
        config = {}
        config['docs'] = [str(doc) for doc in self.docs]
        config['labels'] = self.labels
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
                   config['labels'],
                   model=model,
                   training_config=config['training_config'])
