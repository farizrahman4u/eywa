from eywa.nlu import EntityExtractor

x = ['i live in india', 'i live in china', 'i live in england']
y = ['india', 'china', 'england']

y = [{'place': p} for p in y]

ex = EntityExtractor()
ex.fit(x, y)

for i in x:
    print(ex.predict(i))
