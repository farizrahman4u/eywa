from ..lang import Document

class Memory(object):

    def __init__(self):
        self.docs = []

    def add(self, x):
        if type(x) in (tuple, list):
            self.docs += [Document(i) for i in x]
        else:
            self.docs.append(Document(x))

    def clear(self):
        self.docs = []


    def ask(self, q):
        pass

    def ask_yes_no(self, q):
        pass
