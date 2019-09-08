from ..nlu import Classifier
import json
import responder



class ClassifierServer(object):
    def __init__(self, initial_config=None):
        if initial_config:
            self.load_config(initial_config)
        else:
            self.classifiers = {'default': Classifier()}

    def load_config(self, config):
        if isinstance(config, Classifier):
            self.classifiers = {'default': config}
            return
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
        api = responder.API()
        @api.route('/{classifer}')
        def classify(req, resp, *, classifer):
            clf = self.classifiers.get(classifer)
            if clf is None:
                resp.status_code = 400
                resp.text = "Classifier " + classifer + " not found!"
                return
            text = req.params.get('text')
            print(type(text))
            if text is None:
                data = req.media()
                text = data.get('text')
                if text is None:
                    resp.status_code = 400
                    resp.text = "Required parameter not found: text"
                    return
            return_scores = req.params.get('return_scores', False)
            prediction = clf.predict(text, return_scores=return_scores)
            resp.media = prediction
            resp.status_code = 200
            return
        api.run()
            
            
