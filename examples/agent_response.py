import random
import datetime

class  ResponseClass(object):
  	"""docstring for  ResponseClass"""
  	def __init__(self):
  		super( ResponseClass, self).__init__()

  		
  	def get_response(self, qclass, qentities):

	  	self.responses = { }

	  	i = random.randint(0,2)


	  	if qclass == 'weather':
	  		self.responses['weather' ] = 'The '+qentities['intent']+' in '+qentities['place']+ ' is warm'
	  	if qclass == 'taxi':
	  		self.responses['taxi'    ] =  'Booking a '+qentities['service']+ ' for '+qentities['destination']
	  	if qclass == 'datetime':
	  		self.responses['datetime'    ] =  'Today is '+str(datetime.datetime.today()).split(' ')[0]	  		
	  	if qclass == 'greetings':
	  		self.responses['greetings'] =  ['Hey', 'Hi there', 'Hello'][i] + ['\nwhat would you like me to do ?', '', '\nwhat would you like me to do ?',][i]

	  	return self.responses[qclass]