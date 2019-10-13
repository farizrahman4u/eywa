#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:15:32 2019

@author: abhijithneilabraham
"""

'''
The bot will welcome you with a greeting message once it is running.

current conversatonal capablities.

1. greetings      - Hi , hello etc
2. Booking        - booking service for different vehicles
3. Accomodation   - Accomodation Services
4. Nearby spots   - Looking for nearby locations


Sample conversation of trip booking service

Bot: Hi! What would you like me to do?

Me: Get me a cab.

Bot: Cool! Where would you like to go on the cab?

Me: I would like to go to Chennai.

Bot: How many passengers are going to Chennai?

Me: 4 people

Bot: How long for 4 people to stay?

Me: 5 days
    
Bot: Sure! I'll send you the itineries right away!

Me: Is there bar or pub in the area.

Bot: Looking for bar or pub around the place.
'''

import random
from eywa.nlu import Classifier
from eywa.nlu import EntityExtractor


#Classifier, predicts the class of user input.

CONV_SAMPLES = {
    'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],
    'taxi'      : [' book a bus', 'I need a bus', 'get me a cab'],
    'trip'   : [' I would like to go to NY', 'a trip to Mumbai',
                   'I am going to New York'],
    'seats'  : ['three people', '3 passengers', 'five passengers',
                   '4 people', 'one person'],
    'choice'     : ['find me a restaurant or hotel to have food','beaches or riverside to visit','Find me some hotels or restaurants nearby.'],
    'time':['5 days','5 hours','2 days','2 weeks']
    }
CLF = Classifier()
for key in CONV_SAMPLES:
    CLF.fit(CONV_SAMPLES[key], key)


#Entitiy Extractor, gets the required entities from the input.

X_TRIP = ['I would like to go to Chennai', 'I am going on a trip to Japan',
                   ' to Australia']
Y_TRIP = [{ 'place': 'Chennai'}, { 'place': 'Japan'},
             { 'place': 'Australia'}]

EX_TRIP = EntityExtractor()
EX_TRIP.fit(X_TRIP, Y_TRIP)


X_TAXI = [' get me a cab ', 'need a caravan ', 'find me a cab ',
          'call a taxi ']
Y_TAXI = [{'service': 'cab'}, {'service': 'caravan'},
          {'service': 'cab'},
          {'service': 'taxi'}]

EX_TAXI = EntityExtractor()
EX_TAXI.fit(X_TAXI, Y_TAXI)



X_GREETING = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
Y_GREETING = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'},
              {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]

EX_GREETING = EntityExtractor()
EX_GREETING.fit(X_GREETING, Y_GREETING)


X_SEATS = ['three people', '3 passengers', 'five passengers',
                   '4 people', 'one']
Y_SEATS = [{'number' : 'three'}, {'number' : '3'},{ 'number': 'five'},
              {'number' : '4'}, {'number' : 'one'}]

EX_SEATS = EntityExtractor()
EX_SEATS.fit(X_SEATS, Y_SEATS)
X_CHOICE=['restaurant or hotel to have food','beaches or riverside to visit','any bar or pub nearby']
Y_CHOICE=[{'choice1':'restaurant','choice2':'hotel' },{'choice1':'beaches','choice2':'riverside'},{'choice1':'bar','choice2':'pub'}]
EX_CHOICE=EntityExtractor()
EX_CHOICE.fit(X_CHOICE,Y_CHOICE)
X_TIME=['five days','three hours','two days','four weeks']
Y_TIME=[{'amount':'two','metric':'days'},{'amount':'three','metric':'hours'},{'amount':'two','metric':'days'},{'amount':'four','metric':'weeks'}]
EX_TIME=EntityExtractor()
EX_TIME.fit(X_TIME,Y_TIME)

_EXTRACTORS = {'taxi':EX_TAXI,
               'trip':EX_TRIP,
               'greetings':EX_GREETING,
               'seats':EX_SEATS,
               'choice':EX_CHOICE,
               'time':EX_TIME}


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
    if q_class == 'trip':
        responses['trip'] = 'How many people with you to '+q_entities['place']+"?"


    if q_class == 'taxi':
        responses['taxi'] = 'Cool! where would you like to go on  '+q_entities['service']+"?"

        def get_taxi():
            # Uber/Ola api.
            pass

    if q_class == 'seats':
        responses['seats'] = 'How much time for '+q_entities['number']+ ' people to stay?' 

        def book_seats():
            # seat booking api
            pass

    if q_class == 'greetings':
        responses['greetings'] = ['Hey', 'Hi there',
                                  'Hello'][rd_i]+['\n        what would you like me to do ?', '',
                                                  '\n        what would you like me to do ?'][rd_i]
    if q_class=='choice':
        responses['choice']='Looking for '+q_entities['choice1']+' or a '+q_entities['choice2']+" around the place."
        def map_api():
            pass
    if q_class=='time':
        responses['time']="Sure I'll send the itineries for "+q_entities['amount']+" "+q_entities['metric']+ " of stay right away!"
        def send_itineries():
            pass
     
#        responses['tasks']='You can choose'+q_entities[]
    return 'Agent : '+responses[q_class]



# Conversation loop.

if __name__ == '__main__':
    # Greeting user on startup.
    print(get_response('Hi'))

    while True:
        try:
            UQUERY = input('you   : ')
            if UQUERY == 'bye':
                break
            RESPONSE = get_response(UQUERY)
            print(RESPONSE)
        except:
            print('Agent : sorry wrong input')
