from eywa.nlu import Classifier
import pytest


class TestClassifier:

    def test_classifier_basic(self):
        x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
        x_weather = ['what is the weather like', 'is it hot outside']

        clf = Classifier()
        clf.fit(x_hotel, 'hotel')
        clf.fit(x_weather, 'weather')

        assert clf.predict('will it rain today') == 'weather'
        assert clf.predict('find a place to stay') == 'hotel'

        x_greetings = ['hi', 'hello', 'hello there', 'hey']
        x_not_greetings = x_hotel + x_weather

        clf2 = Classifier()
        clf2.fit(x_greetings, 'greetings')
        clf2.fit(x_not_greetings, 'not_greetings')
        assert clf2.predict('flight information') == 'not_greetings'
        assert clf2.predict('hey there') == 'greetings'

    def test_classifier_serialization(self):
        x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
        x_weather = ['what is the weather like', 'is it hot outside']

        clf1 = Classifier()
        clf1.fit(x_hotel, 'hotel')
        clf1.fit(x_weather, 'weather')

        config = clf1.serialize()
        clf2 = Classifier.deserialize(config)

        assert clf1.classes == clf2.classes

        test_inputs = ['will it rain today', 'find a place to stay']

        for test_input in test_inputs:
            clf1_out = clf1.predict(test_input, return_scores=True)
            clf2_out = clf2.predict(test_input, return_scores=True)
            assert clf1_out == clf2_out



