'''
The bot will welcome you with a greeting message once it is running.

current conversatonal capablities.

1. greetings      - Hi , hello etc
2. cab booking    - book a cab to <place>
3. weather        - what is the weather in <place>
4. date           - what is the date today


Sample conversation

Agent : Hello
        what would you like me to do ?
you   : what is the weather in New York           
Agent : The weather in New_York is warm
you   : book a cab to NYC
Agent : Booking a cab for NYC
you   : what is the date today
Agent : Today is 2019-01-17
'''


from eywa.nlu import Classifier
from eywa.nlu import EntityExtractor
import random
import datetime



'''
Classifer, predicts the class of user input
'''
conv_samples = {
  'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],

  'taxi'      : ['book a cab', 'need a ride', 'find me a cab'],

  'weather'   : ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi'],

  'datetime'  : ['what day is today', 'todays date', 'what time is it now', 'time now', 'what is the time']}

clf  = Classifier()
for key in conv_samples:
    clf.fit(conv_samples[key],key)


'''
Entitiy Extractor, gets the required entities from the input
'''
x_weather = ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi']
y_weather = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'germany'},
                  {'intent': 'weather', 'place': 'kochi'}]

ex_weather = EntityExtractor()
ex_weather.fit(x_weather, y_weather)


x_taxi = ['book a cab to kochi ', 'need a ride to delhi', 'find me a cab for manhattan',
               'call a taxi to calicut']
y_taxi = [ {'service': 'cab', 'destination': 'kochi'}, {'service': 'ride', 'destination' : 'delhi'}, 
                {'service': 'cab', 'destination' : 'manhattan'}, {'service': 'taxi', 'destination' : 'calicut'}]

ex_taxi = EntityExtractor()
ex_taxi.fit(x_taxi, y_taxi)



x_greeting = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
y_greeting = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'}, {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]

ex_greeting = EntityExtractor()
ex_greeting.fit(x_greeting, y_greeting)


x_datetime = ['what day is today', 'date today', 'what time is it now', 'time now']
y_datetime = [ {'intent' : 'day', 'target': 'today'}, {'intent' : 'date', 'target': 'today'},
                    {'intent' : 'time', 'target': 'now'},{'intent' : 'time', 'target': 'now'}]

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

def get_response(uquery):

    responses = { }

    i = random.randint(0,2)

    # Predict the class of the query.
    q_class = clf.predict(uquery) 

    # Run entity extractor of the predicted class on the query.
    q_entities = _extractors[q_class].predict(uquery)


    # Get response based on the class of the query.
    if q_class == 'weather':
        responses['weather' ] = 'The '+q_entities['intent']+' in '+q_entities['place']+ ' is warm'

        def get_weather():
            # Weather api call.
            pass

    if q_class == 'taxi':
        responses['taxi'    ] =  'Booking a '+q_entities['service']+ ' for '+q_entities['destination']

        def get_taxi():
            # Uber/Ola api.
            pass

    if q_class == 'datetime':
        responses['datetime'    ] =  'Today is '+str(datetime.datetime.today()).split(' ')[0]
        def get_dateime():
            # Calender api.
            pass

    if q_class == 'greetings':
        responses['greetings'] =  ['Hey', 'Hi there', 'Hello'][i] + ['\nwhat would you like me to do ?', '', '\n        what would you like me to do ?',][i]

    return 'Agent : '+responses[q_class]

    



'''
Conversation loop
'''

if __name__ == '__main__':
    # Greeting user on startup.
    print(get_response('Hi'))

    # Conversation loop.
    while True:
        uquery = input('you   : ')
        if uquery == 'bye':
            break
        response  = get_response(uquery)
        print(response)