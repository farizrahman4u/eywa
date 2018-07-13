from ..lang import Document



class Pattern(object):

    def __init__(self, pattern, examples={}):
        self.pattern = Document(pattern)
        examples = examples.copy()
        for k in examples:
            v = examples[k]
            if type(v) in (tuple, list):
                examples[k] = [Document(e) for e in v]
            else:
                examples[k] = Document(e)
        self.examples = examples


    def __call__(self, input):
        input = Document(input)
        return {}
