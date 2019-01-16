from eywa.nlu import Classifier
from eywa.nlu import EntityExtractor
import random
import datetime



'''
Classifer for predicting the class of user input
'''

#training data for classifier
conv_samples = {
  'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],

  'taxi'      : ['book a cab', 'need a ride', 'find me a cab'],

  'weather'   : ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi'],

  'datetime'      : ['what day is today', 'todays date', 'what time is it now', 'time now', 'what is the time']
}


# Eywa classifier object
clf  = Classifier()

for key in conv_samples.keys():

	#training the classifier object on the sample inputs
	clf.fit(conv_samples[key],key)


'''
Entitiy Extractor , gets the required entities from the input
'''


#training data for Entitiy Extractor
x_weather = ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi']
y_weather = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'germany'},
                  {'intent': 'weather', 'place': 'kochi'}]

x_taxi = ['book a cab to kochi ', 'need a ride to delhi', 'find me a cab for manhattan',
               'call a taxi to calicut']
y_taxi = [ {'service': 'cab', 'destination': 'kochi'}, {'service': 'ride', 'destination' : 'delhi'}, 
                {'service': 'cab', 'destination' : 'manhattan'}, {'service': 'taxi', 'destination' : 'calicut'}
              ]

x_greeting = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
y_greeting = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'}, {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]

x_datetime = ['what day is today', 'date today', 'what time is it now', 'time now']
y_datetime = [ {'intent' : 'day', 'target': 'today'}, {'intent' : 'date', 'target': 'today'},
                    {'intent' : 'time', 'target': 'now'},{'intent' : 'time', 'target': 'now'}
                  ]



#training the extractors on the data

ex_weather = EntityExtractor()
ex_weather.fit(x_weather, y_weather)


ex_taxi = EntityExtractor()
ex_taxi.fit(x_taxi, y_taxi)

ex_greeting = EntityExtractor()
ex_greeting.fit(x_greeting, y_greeting)

ex_datetime = EntityExtractor()
ex_datetime.fit(x_datetime, y_datetime)


_extractors =  { 'taxi':ex_taxi,
                      'weather':ex_weather, 
                      'greetings':ex_greeting,
                      'datetime':ex_datetime
                    }




'''
Response logic 
'''

def get_response(qclass, qentities):

	responses = { }

	i = random.randint(0,2)


	if qclass == 'weather':
		responses['weather' ] = 'The '+qentities['intent']+' in '+qentities['place']+ ' is warm'

		def get_weather():
			# weather api call
			pass

	if qclass == 'taxi':
		responses['taxi'    ] =  'Booking a '+qentities['service']+ ' for '+qentities['destination']

		def get_taxi():
			# uber/ola api
			pass

	if qclass == 'datetime':
		responses['datetime'    ] =  'Today is '+str(datetime.datetime.today()).split(' ')[0]
		def get_dateime():
			#calender api
			pass

	if qclass == 'greetings':
		responses['greetings'] =  ['Hey', 'Hi there', 'Hello'][i] + ['\nwhat would you like me to do ?', '', '\nwhat would you like me to do ?',][i]

	return responses[qclass]




'''
Conversation loop
'''

if __name__ == '__main__':

	def talker(uquery):

		#predict the class of the query
		q_class = clf.predict(uquery) 

		#run entity extractor of the predicted class on the query
		q_entities = _extractors[q_class].predict(uquery)

		#get response fro the query
		response = get_response(q_class,q_entities)
		

		return response


	print(talker('Hi'))

	while True:

		uquery = input()
		if uquery == 'bye':
			break

		print(talker(uquery))
