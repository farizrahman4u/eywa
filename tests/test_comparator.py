from eywa.nlu import Comparator
import pytest

class TestComparator(object):

	def test_comparator_basic(self):

		examples = [("What is the weather in London", "is London hot today", "where are you from"),
					("What is your name", "Please state your name", "That is a nice place")
					]

		c = Comparator()

		# Score difference check for meaning difference
		for s1,s2,s3 in examples:
			a_score = c(s1,s2) 		
			b_score = c(s1,s3)	
			c_score = c(s2,s1)		

			assert a_score > b_score and a_score == c_score



	def test_comparator_serialization(self):

		examples = [("What is the weather in London", "is London hot today", "where are you from"),
			("What is your name", "Please state your name", "That is a nice place")
			]

		c1 = Comparator()
		config = c1.serialize()

		c2 = Comparator.deserialize(config)

		for s1,s2,s3 in examples:
			c1_score = c1(s1,s2) 		
			c2_score = c2(s1,s2)

			assert c1_score == c2_score




