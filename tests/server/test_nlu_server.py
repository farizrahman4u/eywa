from eywa.nlu import Classifier, EntityExtractor
from eywa.server import NLUServer
import urllib.parse
import json
import time


def test_classifier_server_get():
    x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
    x_weather = ['what is the weather like', 'is it hot outside']
    x_place = ['which place is this', 'where are we', 'where are you going', 'which place is it']
    x_name = ['What is your name', 'Who are you', 'What do i call you', 'what are you called']
    clf = Classifier()
    clf.fit(x_hotel, 'hotel')
    clf.fit(x_weather, 'weather')
    clf.fit(x_place, 'place')
    clf.fit(x_name, 'name')
    clf_serialized = clf.serialize()
    server = NLUServer(clf).serve(test=True)
    r = server.requests.get("/models/classifier/predict?input=will it rain today")
    assert r.text == 'weather'
    r = server.requests.get("/models/classifier/config")
    assert r.json() == clf_serialized


def test_entity_extractor_server_get():
    x = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
    y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]
    ex = EntityExtractor()
    ex.fit(x, y)
    ex_serialized = ex.serialize()
    server = NLUServer(ex).serve(test=True)
    r = server.requests.get("/models/entity_extractor/predict?input=what is the weather in london")
    assert r.json() == {'intent': 'weather', 'place': 'london'}
    r = server.requests.get("/models/entity_extractor/config")
    assert r.json() == json.loads(json.dumps(ex_serialized))


def test_classifier_server_train_get():
    data = {'hotel': ['book a hotel', 'need a nice place to stay', 'any motels near by'],
    'weather': ['what is the weather like', 'is it hot outside'],
    'place': ['which place is this', 'where are we', 'where are you going', 'which place is it'],
    'name': ['What is your name', 'Who are you', 'What do i call you', 'what are you called']
    }
    server = NLUServer().serve(test=True)
    clf = Classifier()
    for y, v in data.items():
       for x in v:
           server.requests.get("/models/classifier/train?input={}&target={}".format(x, y))
           clf.fit(x, y)
    r = server.requests.get("/models/classifier/predict?input=will it rain today")
    assert r.text == 'weather'

def test_classifier_server_train_post():
    data = {'hotel': ['book a hotel', 'need a nice place to stay', 'any motels near by'],
    'weather': ['what is the weather like', 'is it hot outside'],
    'place': ['which place is this', 'where are we', 'where are you going', 'which place is it'],
    'name': ['What is your name', 'Who are you', 'What do i call you', 'what are you called']
    }
    server = NLUServer().serve(test=True)
    clf = Classifier()
    y = tuple(data.keys())
    x = tuple(data.values())
    server.requests.post("/models/classifier/train", json={"data": {"inputs": x, "targets": y}})
    clf.fit(x, y)
    r = server.requests.get("/models/classifier/predict?input=will it rain today")
    assert r.text == 'weather'

def test_entity_extractor_server_train_get():
    X = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
    Y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]
    server = NLUServer().serve(test=True)
    ex = EntityExtractor()
    ex.fit(X, Y)
    for x, y in zip(X, Y):
        r = server.requests.get("/models/entity_extractor/train?input={}&target={}".format(x, urllib.parse.quote(json.dumps(y))))
        assert r.status_code == 200
    test_inp = 'what is the weather in london'
    time.sleep(2)
    r = server.requests.get("/models/entity_extractor/predict?input={}".format(test_inp))
    assert r.json() == ex.predict(test_inp)

def test_entity_extractor_server_train_post():
    X = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
    Y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]
    server = NLUServer().serve(test=True)
    ex = EntityExtractor()
    ex.fit(X, Y)
    r = server.requests.get("/models/entity_extractor/train", json={"data":{"inputs":X, "targets":Y}})
    assert r.status_code == 200
    test_inp = 'what is the weather in london'
    time.sleep(2)
    r = server.requests.get("/models/entity_extractor/predict?input={}".format(test_inp))
    assert r.json() == ex.predict(test_inp)


if __name__ == '__main__':
    pytest.main([__file__])
