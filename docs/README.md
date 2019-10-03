# Eywa - Framework for Conversational Agents

<img  align="right" height="20%" width="20%" src="https://raw.githubusercontent.com/farizrahman4u/eywa/master/logo1.png"/>

[![Build Status](https://travis-ci.org/farizrahman4u/eywa.svg?branch=master)](https://travis-ci.org/farizrahman4u/eywa)

Eywa is an open source framework for building and deploying conversational agents (aka chatbots).

#### Features:
* Requires only few samples for training
* Instant retraining
* Uses word embeddings + heuristics instead of deep learning (better debuggability and interpretability)

## Quickstart
### Classifier

```python
from eywa.nlu import Classifier


x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
x_weather = ['what is the weather like', 'is it hot outside']

clf = Classifier()
clf.fit(x_hotel, 'hotel')
clf.fit(x_weather, 'weather')

print(clf.predict('will it rain today'))  # >>> 'weather'
print(clf.predict('find a place to stay'))  # >>> 'hotel'
```

### Entity extractor

```python
from eywa.nlu import EntityExtractor

x = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]

ex = EntityExtractor()
ex.fit(x, y)

x_test = 'what is the weather in london like'
print(ex.predict(x_test))
```

### Pattern

```python
from eywa.nlu import Pattern

p = Pattern('[fruit: apple, banana] is my favourite fruit')  # create variable [fruit] with sample values {apple, babana}

p('i like grapes')  # >> {'fruit' : 'grapes'}
```

### Requirements

* Python 3.6 or higher
* Eywa requires [Tensorflow 2.0](https://www.tensorflow.org/install/pip) and should be installed manually by the user (is not installed automatically as a dependency)


## Installation

### Via pip:

`pip install eywa`

### Install from source:

```
git clone https://www.github.com/farizrahman4u/eywa.git
cd eywa
python setup.py install
```
