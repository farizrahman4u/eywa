from eywa.nlu import Pattern
import pytest

def test_pattern():

    p = Pattern('[fruit : apple, banana] is my favourite fruit')
    assert p(' i like grape') == {'fruit' : 'grape'}

def test_pattern_serialization():

    p1 = Pattern('[fruit : apple, banana] is my favourite fruit')
    p1(' i like grape') 

    config = p1.serialize()
    p2 = Pattern.deserialize(config)

    test_inputs = ['i like grape']

    for test_input in test_inputs:
        p1_out = p1(test_input)
        p2_out = p2(test_input)
        assert p1_out == p2_out


if __name__ == '__main__':
    pytest.main(__file__)
