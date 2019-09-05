from ..nlu import Classifier
import json

class ClassifierServer(object):
    def __init__(self, initial_config=None):
        if initial_config:
            self.load_config(initial_config)
        else:
            self.classifiers = {'default_classifier': Classifier()}

    def load_config(self, config):
        if isinstance(config, str):
            config = json.loads(config)
        assert isinstance(config, dict)
        clf_configs = config.get('classifiers')
        assert isinstance(clf_configs, dict)
        for clf_name in clf_configs:
            assert isinstance(clf_name, str)
            clf_config = clf_configs[clf_name]
            assert isinstance(clf_config, dict)
            clf = Classifier.deserialize(clf_config)
            self.classifiers[clf_name] = clf

    def get_config(self):
        clfs_config = {}
        for name, clf in self.classifiers.items():
            clfs_config[name] = clf.serialize()
        config = {'classifiers': clfs_config}
        return config

    def train(classifier_name, inputs, targets):
        clf = self.classifiers.get(classifier_name)
        assert clf is not None
        if isinstance(inputs, list):
            for inp in inputs:
                assert isinstance(inp, str)
        else:
            assert isinstance(inputs, str)
        if isinstance(targets, list):
            for t in inputs:
                assert isinstance(t, str)
        else:
            assert isinstance(targets, str)
        clf.fit(inputs, targets)

    def predict(self, inputs):
        pass

    def serve(self, port=None):
        pass
