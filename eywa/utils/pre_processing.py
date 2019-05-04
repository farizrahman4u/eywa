import json
import ast

class Preprocessing(object):
    '''
    Processing training data into required format
    
    1. X - text, Y - entities, for entitiy extractor
    '''

    def __init__(self):
        
        pass



    @classmethod
    def prep_entity_extractor_data(self, filename):
        '''
        converts txt data to entity extractor format
        '''
        if not filename.endswith('.txt') :
            print("Error : Please ensure the file is in txt format")   
            exit()

        with open(filename, "r") as read_file:
            data = [line.rstrip('\n') for line in read_file.readlines()]

        X = []
        Y = []
        for line in data:
            if line.startswith('#'):
                X.append(line[1:])
            elif line.startswith('-'):
                Y.append(ast.literal_eval(line[1:]))
            else:
                pass
        return X,Y

    
    def preprocess(self, filename=None, prep_for=None): 

        if filename is None:
            print("Error : Please enter a valid filename")
            exit()
        if prep_for is None:
            print("Error : Please provide the unit for whihc data to prepare\n1.entity_extractor")
            exit()
        if prep_for == 'entity_extractor':
            return self.prep_entity_extractor_data(filename)