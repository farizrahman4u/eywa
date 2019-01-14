from agent_classification import ClassifierClass
from agent_entity_extractor import EntityClass
from agent_response import ResponseClass

e = EntityClass()
ex = e.get_extractors()

c = ClassifierClass()
cx = c.get_classifier()



r = ResponseClass()
rx = r.get_response

samples = ['get me a taxi to germany' , 'what is it like in chennai']


def talker(uquery):

	#predict the class of the query
	q_class = cx.predict(uquery) 

	#run entity extractor of the predicted class on the query
	q_entities = ex[q_class].predict(uquery)

	#get response fro the query
	response = rx(q_class,q_entities)
	

	return response

print(talker('Hi'))

while True:

	uquery = input()
	if uquery == 'bye':
		break

	print(talker(uquery))