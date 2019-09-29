from ..nlu import Classifier, EntityExtractor
from ..utils import server_dir
import json
import responder
import asyncio
import os

    

def _get_media(req):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        med = loop.run_until_complete(req.media())
        loop.close()
        return med
    except:
        return {}



class NLUServer(object):
    def __init__(self, config=None, path=None, name=None, auto_save=True):
        s = set([config, path, name])
        s.remove(None)
        if len(s) > 1:
            raise Exception("More than 1 of config, path and name args were provided.")
        if config:
            self.load_config(config)
            self.path = None
        elif path:
            with open(path, 'r') as f:
                self.load_config(json.load(f))
            self.path = path
        else:
            self.models = {
                'classifier': Classifier(),
                'entity_extractor': EntityExtractor()
            }
            if name:
                self.path = os.path.join(server_dir, name) + '.json'
                self.save()
            else:
                self.path = None
        self.auto_save = auto_save

    def save(self, path=None):
        if path is None:
            path = self.path
            if path is None:
                raise Exception('Path not provided!')
        with open(path, 'w') as f:
            json.write(self.get_config(), f)

    def load_config(self, config):
        if isinstance(config, Classifier):
            self.models = {
                'classifier': config
            }
            return
        elif isinstance(config, EntityExtractor):
            self.models = {
                'entity_extractor': config
            }
            return
        if isinstance(config, str):
            config = json.loads(config)
        assert isinstance(config, dict)
        self.models = {}
        models_config = config.get('models')
        assert isinstance(models_config, dict)
        for k, v in models_config.items():
            assert isinstance(k, str)
            cls_name = v.get('class_name')
            if cls_name == 'Classifier':
                self.models[k] = Classifier.deserialize(v)
            elif cls_name == 'EntityExtractor':
                self.models[k] = EntityExtractor.deserialize(v)
            elif cls_name is None:
                raise Exception('Key class_name not found in config.')
            else:
                raise Exception("Invalid model type: " + str(cls_name))

    def get_config(self):
        models_config = {}
        for name, model in self.models.items():
            models_config[name] = model.serialize()
        config = {'models': models_config}
        return config

    def _verify_train(self, model, inputs, targets):
        if isinstance(model, Classifier):
            if isinstance(inputs, list):
                for inp in inputs:
                    assert isinstance(inp, str)
            else:
                assert isinstance(inputs, str)
            if isinstance(targets, list):
                for t in targets:
                    assert isinstance(t, str)
            else:
                assert isinstance(targets, str)
        elif isinstance(model, EntityExtractor):
            if isinstance(inputs, list):
                for inp in inputs:
                    assert isinstance(inp, str)
            else:
                assert isinstance(inputs, str)
            if isinstance(targets, list):
                for t in targets:
                    assert isinstance(t, dict)
                    for k, v in t.items():
                        assert isinstance(k, str)
                        assert isinstance(v, (str, type(None)))
            else:
                assert isinstance(targets, dict)
                for k, v in targets.items():
                    assert isinstance(k, str)
                    assert isinstance(v, (str, type(None)))
        if self.auto_save and self.path:
            self.save()

    def serve(self, port=None, test=False):
        api = responder.API()
        @api.route('/models/{model_name}/predict')
        def predict(req, resp, *, model_name):
            model = self.models.get(model_name)
            if model is None:
                resp.status_code = 400
                resp.text = "Model " + model_name + " not found!"
                return
            inp = req.params.get('input')
            med = _get_media(req)
            if inp is None:
                inp = med.get('input')
                if inp is None:
                    resp.status_code = 400
                    resp.text = "No input provided!"
                    return
            return_scores = req.params.get('return_scores')
            if return_scores is None:
                return_scores = med.get('return_scores', False)
            out = model.predict(inp, return_scores=return_scores)
            if isinstance(out, str):
                resp.text = out
            else:
                resp.media = out
            resp.status_code = 200
        @api.route('/models/{model_name}/train')
        def train(req, resp, *, model_name):
            model = self.models.get(model_name)
            if model is None:
                resp.status_code = 400
                resp.text = "Model " + model_name + " not found!"
                return
            inp = req.params.get('input')
            if inp is None:
                med = _get_media(req)
                data = med.get('data')
                if data is None:
                    resp.status_code = 400
                    resp.text = "No data provided for training!"
                inp = data.get('inputs')
                if inp is None:
                    resp.status_code = 400
                    resp.text = "Key error: inputs"
                targ = data.get('targets')
                if targ is None:
                    resp.status_code = 400
                    resp.text = "Key error: targets"                    
            else:
                targ = req.params.get('target')
                if isinstance(model, EntityExtractor):
                    try:
                        targ = json.loads(targ)
                    except:

                        resp.status_code = 400
                        resp.text = "Invalid target json."
                        return
                if targ is None:
                    resp.status_code = 400
                    resp.text = "Key error: target"
            try:
                self._verify_train(model, inp, targ)
                resp.status_code = 200
                resp.media = {"STATUS": "OK"}
            except Exception as e:
                resp.status_code = 500
                resp.text = "Training data verification failed: " + str(e)
            @api.background.task
            def train_bg():
                model.fit(inp, targ)
            train_bg()
        @api.route('/config')
        def config(req, resp, *args):
            resp.status_code = 200
            resp.media = self.get_config()

        @api.route('/models/{model_name}/config')
        def models_config(req, resp, *, model_name):
            model = self.models.get(model_name)
            if model is None:
                resp.status_code = 400
                resp.text = "Model " + model_name + " not found!"
                return
            resp.status_code = 200
            resp.media = model.serialize()

        @api.route('/load_config')
        def load_config(req, resp, *args):
            med = _get_media(req)
            config = med.get('config')
            if config is None:
                resp.status_code = 400
                resp.text = "Config not provided!"
            try:
                self.load_config(config)
                resp.status_code = 200
                resp.text = "Config loaded successfully!"
            except Exception as e:
                resp.status_code = 500
                resp.text = "Error loading config: " + str(e)

        @api.route('/add_classifier/{clf_name}')
        def add_classifer(req, resp, *, clf_name):
            if clf_name in self.models:
                resp.status_code = 500
                resp.text = "Model with name " + clf_name + "already exists!"
            self.models[clf_name] = Classifier()
            resp.status_code = 500
            resp.text = "Classifier added: " + clf_name
            if self.auto_save and self.path:
                @api.background.task
                def save_bg():
                    self.save()
                save_bg()

        @api.route('/add_entity_extractor/{ex_name}')
        def add_entity_extractor(req, resp, *, ex_name):
            if ex_name in self.models:
                resp.status_code = 500
                resp.text = "Model with name " + ex_name + "already exists!"
                return
            self.models[ex_name] = EntityExtractor()
            resp.status_code = 200
            resp.text = "EntityExtractor added: " + ex_name
            if self.auto_save and self.path:
                @api.background.task
                def save_bg():
                    self.save()
                save_bg()

        @api.route('/models')
        def get_models(req, resp, *args):
            models = []
            for k, v in self.models.items():
                models.append({'name': k, 'type': v.__class__.__name__})
            resp.status_code = 200
            resp.media = models

        @api.route('/models/{model_name}/delete')
        def delete_model(req, resp, *, model_name):
            model = self.models.get(model_name)
            if model is None:
                resp.status_code = 400
                resp.text = "Model " + model_name + " not found!"
                return
            del self.models[model_name]
            resp.status_code = 500
            resp.text = "Model delted: " + model_name
            if self.auto_save and self.path:
                @api.background.task
                def save_bg():
                    self.save()
                save_bg()
        if test:
            return api
        else:
            api.run(port=port)
