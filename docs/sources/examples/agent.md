
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


```python
import random
import datetime
from eywa.nlu import Classifier
from eywa.nlu import EntityExtractor


# Classifier, predicts the class of user input.

CONV_SAMPLES = {
    'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],
    'taxi'      : ['book a cab', 'need a ride', 'find me a cab'],
    'weather'   : ['what is the weather in tokyo', 'weather germany',
                   'what is the weather like in kochi'],
    'datetime'  : ['what day is today', 'todays date', 'what time is it now',
                   'time now', 'what is the time']}

CLF = Classifier()
for key in CONV_SAMPLES:
    CLF.fit(CONV_SAMPLES[key], key)



# Entitiy Extractor, gets the required entities from the input.

X_WEATHER = ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi']
Y_WEATHER = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'germany'},
             {'intent': 'weather', 'place': 'kochi'}]

EX_WEATHER = EntityExtractor()
EX_WEATHER.fit(X_WEATHER, Y_WEATHER)


X_TAXI = ['book a cab to kochi ', 'need a ride to delhi', 'find me a cab for manhattan',
          'call a taxi to calicut']
Y_TAXI = [{'service': 'cab', 'destination': 'kochi'}, {'service': 'ride', 'destination' : 'delhi'},
          {'service': 'cab', 'destination' : 'manhattan'},
          {'service': 'taxi', 'destination' : 'calicut'}]

EX_TAXI = EntityExtractor()
EX_TAXI.fit(X_TAXI, Y_TAXI)



X_GREETING = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
Y_GREETING = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'},
              {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]

EX_GREETING = EntityExtractor()
EX_GREETING.fit(X_GREETING, Y_GREETING)


X_DATETIME = ['what day is today', 'date today', 'what time is it now', 'time now']
Y_DATETIME = [{'intent' : 'day', 'target': 'today'}, {'intent' : 'date', 'target': 'today'},
              {'intent' : 'time', 'target': 'now'}, {'intent' : 'time', 'target': 'now'}]

EX_DATETIME = EntityExtractor()
EX_DATETIME.fit(X_DATETIME, Y_DATETIME)


_EXTRACTORS = {'taxi':EX_TAXI,
               'weather':EX_WEATHER,
               'greetings':EX_GREETING,
               'datetime':EX_DATETIME}


# Response logic.

def get_response(u_query):
    '''
    Accepts user query and returns a response based on the class of query
    '''
    responses = {}
    rd_i = random.randint(0, 2)

    # Predict the class of the query.
    q_class = CLF.predict(u_query)

    # Run entity extractor of the predicted class on the query.
    q_entities = _EXTRACTORS[q_class].predict(u_query)


    # Get response based on the class of the query.
    if q_class == 'weather':
        responses['weather'] = 'The '+q_entities['intent']+' in '+q_entities['place']+ ' is warm'

        def get_weather():
            # Weather api call.
            pass

    if q_class == 'taxi':
        responses['taxi'] = 'Booking a '+q_entities['service']+ ' for '+q_entities['destination']

        def get_taxi():
            # Uber/Ola api.
            pass

    if q_class == 'datetime':
        responses['datetime'] = 'Today is '+str(datetime.datetime.today()).split(' ')[0]

        def get_dateime():
            # Calender api.
            pass

    if q_class == 'greetings':
        responses['greetings'] = ['Hey', 'Hi there',
                                  'Hello'][rd_i]+['\n        what would you like me to do ?', '',
                                                  '\n        what would you like me to do ?'][rd_i]

    return 'Agent : '+responses[q_class]



# Conversation loop.

if __name__ == '__main__':
    # Greeting user on startup.
    print(get_response('Hi'))

    while True:
        UQUERY = input('you   : ')
        if UQUERY == 'bye':
            break
        RESPONSE = get_response(UQUERY)
        print(RESPONSE)
```