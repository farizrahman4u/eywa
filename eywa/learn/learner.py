class Learner(object):

    def __init__(self):
        pass

    def __call__(self, function, weights, inputs, outputs, loss='mse'):
        # Arguments
        # function : The function to be evaluated. Callable.
        # weights : list of floats
        # inputs : list
        # outputs : list, same length as inputs
        # loss : Loss function to compute distance between prediction and truth
        raise NotImplemented
