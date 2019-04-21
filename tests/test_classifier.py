from eywa.nlu import Classifier
import pytest


class TestClassifier:

    def test_classifier_basic(self):
        x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
        x_weather = ['what is the weather like', 'is it hot outside']
        x_place = ['which place is this', 'where are we', 'where are you going', 'which place is it']
        x_name = ['What is your name', 'Who are you', 'What do i call you', 'what are you called']


        clf = Classifier()
        clf.fit(x_hotel, 'hotel')
        clf.fit(x_weather, 'weather')
        clf.fit(x_place, 'place')
        clf.fit(x_name, 'name')


        assert clf.predict('will it rain today') == 'weather'
        assert clf.predict('find a place to stay') == 'hotel'
        assert clf.predict('where am I') == 'place'
        assert clf.predict('may i know your name') == 'name'

        # Asserting return probablities for each of the above tests
        assert clf.predict('will it rain today', True) == {'hotel': 0.4461936950683594, 'weather': 0.45663195848464966, 'place': 0.4499515891075134, 'name': 0.4477596879005432}
        assert clf.predict('find a place to stay', True) == {'hotel': 0.49977487325668335, 'weather': 0.4606366753578186, 'place': 0.47342991828918457, 'name': 0.4549995958805084}
        assert clf.predict('where am I',True) == {'hotel': 0.45000576972961426, 'weather': 0.4549625515937805, 'place': 0.46232348680496216, 'name': 0.4510326385498047}
        assert clf.predict('may i know your name', True) == {'hotel': 0.46361255645751953, 'weather': 0.46408894658088684, 'place': 0.4642796516418457, 'name': 0.47779345512390137}        


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


if __name__ == '__main__':
    pytest.main([__file__])
