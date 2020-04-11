from eywa.nlu import EntityExtractor
import pytest



def test_entity_extractor_basic():
    x = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
    y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]

    ex = EntityExtractor()
    ex.fit(x, y)

    x_test = 'what is the weather like in london'
    assert ex.predict(x_test) == {'intent': 'weather', 'place': 'london'}

def test_entity_extractor_serialization():
    x = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
    y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]

    ex1 = EntityExtractor()
    ex1.fit(x, y)

    config = ex1.serialize()
    ex2 = EntityExtractor.deserialize(config)

    test_inputs = ['what is the weather in london like']

    for test_input in test_inputs:
        ex1_out = ex1.predict(test_input)
        ex2_out = ex2.predict(test_input)
        ex2.predict(test_input, return_scores=True) # TODO
        assert ex1_out == ex2_out


if __name__ == '__main__':
    pytest.main([__file__])
