from eywa.nlu import EntityExtractor


class EntityClass(object):

    def __init__(self):

        self.x_weather = ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi']
        self.y_weather = [{'intent': 'weather', 'place': 'tokyo'}, {'intent': 'weather', 'place': 'germany'},
                          {'intent': 'weather', 'place': 'kochi'}]

        self.x_taxi = ['book a cab to kochi ', 'need a ride to delhi', 'find me a cab for manhattan',
                       'call a taxi to calicut']
        self.y_taxi = [ {'service': 'cab', 'destination': 'kochi'}, {'service': 'ride', 'destination' : 'delhi'}, 
                        {'service': 'cab', 'destination' : 'manhattan'}, {'service': 'taxi', 'destination' : 'calicut'}
                      ]

        self.x_greeting = ['Hii', 'helllo', 'Howdy', 'hey there', 'hey', 'Hi']
        self.y_greeting = [{'greet': 'Hii'}, {'greet': 'helllo'}, {'greet': 'Howdy'}, {'greet': 'hey'}, {'greet': 'hey'}, {'greet': 'Hi'}]

        self.x_datetime = ['what day is today', 'date today', 'what time is it now', 'time now']
        self.y_datetime = [ {'intent' : 'day', 'target': 'today'}, {'intent' : 'date', 'target': 'today'},
                            {'intent' : 'time', 'target': 'now'},{'intent' : 'time', 'target': 'now'}
                          ]



        self.ex_weather = EntityExtractor()
        self.ex_weather.fit(self.x_weather, self.y_weather)


        self.ex_taxi = EntityExtractor()
        self.ex_taxi.fit(self.x_taxi, self.y_taxi)

        self.ex_greeting = EntityExtractor()
        self.ex_greeting.fit(self.x_greeting, self.y_greeting)

        self.ex_datetime = EntityExtractor()
        self.ex_datetime.fit(self.x_datetime, self.y_datetime)

        self._EXTRACTORS =  { 'taxi':self.ex_taxi,
                              'weather':self.ex_weather, 
                              'greetings':self.ex_greeting,
                              'datetime':self.ex_datetime
                            }


    def get_extractors(self):
        return self._EXTRACTORS




