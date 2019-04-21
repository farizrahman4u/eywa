from eywa.nlu import Classifier
import pytest


class TestClassifier:

    def test_classifier_basic(self):
        x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
        x_weather = ['what is the weather like', 'is it hot outside']
        x_place = ['which place is this', 'where are we', 'where are you going', 'which place is it']
        x_name = ['What is your name', 'Who are you', 'What do i call you', 'what are you called']
        
        clf1 = Classifier()
        clf2 = Classifier()
        clfs = [clf1, clf2]
        for cl in clfs:
            cl.fit(x_hotel, 'hotel')
            cl.fit(x_weather, 'weather')
            cl.fit(x_place, 'place')
            cl.fit(x_name, 'name')

        assert clf1.predict('will it rain today') == 'weather'
        assert clf1.predict('find a place to stay') == 'hotel'
        assert clf1.predict('where am I') == 'place'
        assert clf1.predict('may i know your name') == 'name'

        # Testing return probablities for each of the above tests
        assert clf1.predict('will it rain today', True) == clf2.predict('will it rain today', True)
        assert clf1.predict('find a place to stay', True) == clf2.predict('find a place to stay', True)
        assert clf1.predict('where am I',True) == clf2.predict('where am I',True)
        assert clf1.predict('may i know your name', True) == clf2.predict('may i know your name', True)


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


if __name__ == '__main__':
    pytest.main([__file__])
