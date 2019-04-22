from eywa.nlu import EntityExtractor
import pytest


class TestEntityExtractor:

    def test_entity_extractor_basic(self):
        x_weather = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi', 'weather here']
        y_weather = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': None}, {'intent': 'weather', 'place': 'kochi'}, {'intent':'weather' , 'place': None}]

        ex = EntityExtractor()
        ex.fit(x_weather, y_weather)

        x_test = 'what is the weather in london like'
        assert ex.predict(x_test) == {'intent': 'weather', 'place': 'london'}
        x_test = 'how is the weather'
        assert ex.predict(x_test) == {'intent': 'weather', 'place': None}

        # Tests with keys .

        x_test = 'Hows the weather in yokohama'
        assert ex.predict(x_test, ['place']) == {'place': 'yokohama'}

    def test_entity_extractor_serialization(self):
        x_weather = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi', 'weather here']
        y_weather = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': None}, {'intent': 'weather', 'place': 'kochi'}, {'intent':'weather' , 'place': None}]

        ex1 = EntityExtractor()
        ex1.fit(x_weather, y_weather)

        config = ex1.serialize()
        ex2 = EntityExtractor.deserialize(config)

        test_inputs = ['what is the weather in london like', 'weather now', 'tommorows weather']

        for test_input in test_inputs:
            ex1_out = ex1.predict(test_input)
            ex2_out = ex2.predict(test_input)
            assert ex1_out == ex2_out
            ex1_out = ex1.predict(test_input, return_scores=True)
            ex2_out = ex2.predict(test_input, return_scores=True)
            assert ex1_out == ex2_out

if __name__ == '__main__':
    pytest.main([__file__])
