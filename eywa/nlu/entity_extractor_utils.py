"""Checks, utilities for entity_extractor
"""

def check_inputs_targets_consistency(inputs, targets):
    if type(inputs) != list:
        raise TypeError('Input must be of type list.' 
                        'Got %s instead'%type(inputs))
    if type(targets) != list:
        raise TypeError('Target must be of type list.' 
                        'Got %s instead'%type(targets))
    if len(inputs) != len(targets):
        raise ValueError('Inputs should have the same'
                        ' number of samples as targets.'
                        'Found ' + str(len(inputs)) +
                        ' input samples and ' + str(len(targets)) +
                        ' target samples')


