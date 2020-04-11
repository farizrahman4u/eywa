from eywa.nn import NNClassifier
import pytest
from eywa.nn import NNClassifier
import pytest

def test_nn_classifier_basic():
    
    docs=['book for a holiday place','any nice places to spend the night','rent a hotel room','find a place to stay','what is the weather in kochi','weather america','will it snow today','it is a sunny day']
    labels=['hotel','hotel','hotel','hotel','weather','weather','weather','weather']
    nnclf=NNClassifier(docs, labels)
    x_tests = ['book for a place to stay',
                'rent a holiday place','weather germany','will it rain today']
    y_tests = ['hotel','hotel','weather','weather']
    for x,y in zip(x_tests, y_tests):
        assert nnclf.predict(x) == y
def test_nn_classifier_serialize():
     
    docs=['book for a holiday place','any nice places to spend the night','rent a hotel room','find a place to stay','what is the weather in kochi','weather america','will it snow today','it is a sunny day']
    labels=['hotel','hotel','hotel','hotel','weather','weather','weather','weather']
    nnclf1 = NNClassifier(docs, labels)
    
    config  =  nnclf1.serialize()
    nnclf2 = NNClassifier.deserialize(config)

    assert nnclf1.values == nnclf2.values

    x_tests = ['book for a place to stay',
                'rent a holiday place','weather germany','will it rain today']
    for x in x_tests:
        assert nnclf1.predict(x) == nnclf2.predict(x)
    
if __name__ == '__main__':
    pytest.main([__file__])