from eywa.nn import NNPicker
import pytest


def test_nn_picker_basic():
    docs = [
        'get me a cab from Aluva to kakkanad for 2 at 2am today',
        'find me a txi from manglore to mumbai for 5 at 6pm tommorow',
        'book a cab to newyork for 5',
        'find a taxi from akihabara to yokohama'
    ]
    values = [
        {'intent': 'cab_booking', 'source': 'Aluva', 'destination': 'kakkanad', 'nop': '2', 'time': '2am', 'date': 'today'},
        {'intent': 'cab_booking', 'source': 'manglore', 'destination': 'mumbai', 'nop': '5', 'time': '6pm', 'date': 'tommorow'},
        {'intent': 'cab_booking', 'source': None, 'destination': 'newyork', 'nop': '5', 'time': None, 'date': None},
        {'intent': 'cab_booking', 'source': 'akihabara', 'destination': 'yokohama', 'nop': None, 'time': None, 'date': None}
    ]

    nnpicker = NNPicker(docs, values)

    x_tests = ['get me a taxi from chennai to banglore for 2 at 2am today',
                'get me a taxi from chennai to banglore for 2am']
    y_tests = [{'source': 'chennai', 'destination': 'banglore', 'nop': '2', 'time': '2am', 'date': 'today'},
                {'source': 'chennai', 'destination': 'banglore', 'time': '2am'}]
    
    for x,y in zip(x_tests, y_tests):
        assert nnpicker.predict(x) == y

def test_nn_picker_serialize():
    docs = [
        'get me a cab from Aluva to kakkanad for 2 at 2am today',
        'find me a txi from manglore to mumbai for 5 at 6pm tommorow',
        'book a cab to newyork for 5',
        'find a taxi from akihabara to yokohama'
    ]
    values = [
        {'intent': 'cab_booking', 'source': 'Aluva', 'destination': 'kakkanad', 'nop': '2', 'time': '2am', 'date': 'today'},
        {'intent': 'cab_booking', 'source': 'manglore', 'destination': 'mumbai', 'nop': '5', 'time': '6pm', 'date': 'tommorow'},
        {'intent': 'cab_booking', 'source': None, 'destination': 'newyork', 'nop': '5', 'time': None, 'date': None},
        {'intent': 'cab_booking', 'source': 'akihabara', 'destination': 'yokohama', 'nop': None, 'time': None, 'date': None}
    ]

    nnpicker1 = NNPicker(docs, values)
    
    config  =  nnpicker1.serialize()
    nnpicker2 = NNPicker.deserialize(config)

    assert nnpicker1.values == nnpicker2.values

    x_tests = ['get me a taxi from chennai to banglore for 2 at 2am today',
                'get me a taxi from chennai to banglore for 2am']

    for x in x_tests:
        assert nnpicker1.predict(x) == nnpicker2.predict(x)

if __name__ == '__main__':
    pytest.main([__file__])
