from eywa.nlu import Classifier
from eywa.lang import *


x_hotel = ['book a hotel', 'need a nice place to stay', 'any motels near by']
x_weather = ['what is the weather like', 'is it hot outside']

clf = Classifier()
clf.fit(x_hotel, 'hotel')
clf.fit(x_weather, 'weather')

print(clf.predict('will it rain today'))
print(clf.predict('find a place to stay'))


x_greetings = ['hi', 'hello', 'hello there', 'hey']
x_not_greetings = x_hotel + x_weather

clf = Classifier()
clf.fit(x_greetings, 'greetings')
clf.fit(x_not_greetings, 'not_greetings')
print(clf.predict('flight information'))
print(clf.predict('hey there'))
