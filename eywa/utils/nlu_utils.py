"""
Checks, utilities for nlu module
"""

def check_inputs_targets_consistency(inputs, targets, nlu_type):
    if nlu_type == "entity_extractor":
        input_len = len(inputs)
        target_len = len(targets)
        if type(inputs) == str:
            input_len = 1
        if type(targets) == dict:
            target_len = 1
        if type(inputs) not in (list, str):
            raise TypeError('Input must be list of strings or single' 
                            'string.Got %s instead'%type(inputs))
        if type(targets) not in (list, dict):
            raise TypeError('Target must be list of dicts or single' 
                            'dict .Got %s instead'%type(targets))
        if input_len != target_len:
            raise ValueError('Inputs should have the same'
                            ' number of samples as targets.'
                            'Found ' + str(input_len) +
                            ' input samples and ' + str(target_len) +
                            ' target samples')

    elif nlu_type == "classifier":
        if len(inputs) != len(targets):
            raise ValueError('Inputs should have the same'
                            ' number of samples as targets.'
                            'Found ' + str(len(inputs)) +
                            ' input samples and ' + str(len(targets)) +
                            ' target samples')
