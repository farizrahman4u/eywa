from eywa.nlu import Classifier


class ClassifierClass(object):


	def __init__(self):

		self.clf  = Classifier()

		self.conv_samples = {
		  'greetings' : ['Hi', 'hello', 'How are you', 'hey there', 'hey'],

		  'taxi'      : ['book a cab', 'need a ride', 'find me a cab'],

		  'weather'   : ['what is the weather in tokyo', 'weather germany', 'what is the weather like in kochi'],

		  'datetime'      : ['what day is today', 'todays date', 'what time is it now', 'time now', 'what is the time']
		}


		for key in self.conv_samples.keys():
			self.clf.fit(self.conv_samples[key],key)


	def get_classifier(self):
		return self.clf
