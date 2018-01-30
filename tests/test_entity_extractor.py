from eywa.nlu import EntityExtractor

x = ['what is the weather in tokyo', 'what is the weather', 'what is the weather like in kochi']
y = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'here'}, {'intent': 'weather', 'place': 'kochi'}]

ex = EntityExtractor()
ex.fit(x, y)

x_test = 'what is the weather in london like'
print(ex.predict(x_test))
