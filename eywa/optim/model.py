class Model(object):

    def __init__(self):
        if not hasattr(self, 'weights'):
            self.weights = []

    def __call__(self, x):
        return x
