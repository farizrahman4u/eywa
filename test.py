from eywa.nlu import EntityExtractor
import pytest

X = [['what is the weather in tokyo', 'what is the weather'], 
    'what is the weather like in kochi', 
    ['what is the weather like in hawaii', 'what is the weather in aluva']]

Y_True = [[{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}],
        {'intent': 'weather', 'place': 'kochi'}, 
        [{'intent': 'weather', 'place': 'hawaii'}, {'intent': 'weather', 'place': 'aluva'}]]

Y_False = [[{'intent': 'weather', 'place': 'tokyo'}], 
          [{'intent': 'weather', 'place': 'kochi'},{'intent': 'weather'}],
          {'intent': 'weather', 'place': 'aluva'}]


ex1 = EntityExtractor()
for x,y in zip(X, Y_True):
    ex1.fit(x,y)

ex1 = EntityExtractor()
for x,y in zip(X, Y_False):
    with pytest.raises(ValueError, match=r"NLU:.*"):
        ex1.fit(x,y)