from .graph import Node
from .blame import Blame, BlameType


class Select(Node):
    def __init__(self, name=None):
        super(Select, self).__init__(name)
        self.add_input('selector')

    def f(self, inputs):
        assert 'selector' in inputs
        selected_input = inputs['selector']
        assert selected_input != 'selector'
        return inputs[selected_input]

    def add_input(self, name):
        if name == 'selector' and name in self.input_nodes:
            # can happen while deserialization
            return
        super(Select, self).add_input(name)

    def inputs_ready(self):  # we dont need all inputs
        s_value = self.input_values['selector']
        if s_value is None:
            return False
        inp_val = self.input_values[s_value]
        if inp_val is None:
            return False
        return True

    def all_inputs_ready(self):
        return super(Select, self).inputs_ready()

    def pull(self):  # we dont need all inputs
        if self.cache and self.value is not None and not self._inputs_changed:
            return self.value
        s_node = self.input_nodes['selector']
        if s_node is None:
            s_val = self.input_values['selector']
            if s_val is None:
                raise Exception('Value not provided for selector input.')
        else:
            s_val = s_node.pull()
            self.input_values['selector'] = s_val
        inp_node = self.input_nodes[s_val]
        if inp_node is None:
            inp_val = self.input_values[s_val]
            if inp_val is None:
                raise Exception("Value not provided for input node " + s_val)
        else:
            self.input_values[s_val] = inp_node.pull()
        self.run()
        return self.value

    def pull_all(self):
        return super(Select, self).pull()
    
    def inrange(self, y):
        for name, node in self.input_nodes.items():
            if name == 'selector':
                continue
            if node is not None and node.inrange(y):
                True
        return False

    def blame(self, blame):
        super(Select, self).blame(blame)
        if not self.all_inputs_ready():
            self.pull_all()
        if blame.blame_type == BlameType.POSITIVE:
            self.input_nodes['selector'].blame(blame.fork())
            self.input_nodes[self.input_values['selector']].blame(blame.fork())
        elif blame.blame_type == BlameType.CORRECTIVE:
            for name, val in self.input_values:
                if name == 'selector':
                    continue
                if val == blame.expected:
                    if name == self.input_values['selector']:
                        self.input_nodes['selector'].blame(blame.fork(BlameType.POSITIVE))
                    else:
                        self.input_nodes['selector'].blame(blame.fork(BlameType.CORRECTIVE, name))
                    self.input_nodes[name].blame(blame.fork(BlameType.POSITIVE))
                    return
            for name, node in self.input_nodes:
                if node is None:
                    continue
                if name == 'selector':
                    continue
                if node.inrange(blame.expected):
                    if name == self.input_values['selector']:
                        self.input_nodes['selector'].blame(blame.fork(BlameType.POSITIVE))
                    else:
                        return self.input_nodes['selector'].blame(blame.fork(BlameType.CORRECTIVE, name))
                    node.blame(blame.fork())
        elif blame.blame_type == BlameType.NEGATIVE:
            c = blame.confidence * 0.5
            self.input_nodes['selector'].blame(blame.fork(confidence=c))
            self.input_nodes[self.input_values['selector']].blame(blame.fork(confidence=c))
            
