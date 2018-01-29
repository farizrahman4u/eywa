from eywa.nlu import EntityExtractor

x = ['i live in india', 'i kind of live in china', 'i live in england', 'england is where i live']
y = ['india', 'china', 'england', 'england']

y = [{'place': p} for p in y]

ex = EntityExtractor()
ex.fit(x, y)
x2 = ['paris is where i live']
for i in x2:
    print(ex.predict(i))
