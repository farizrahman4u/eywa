from .signal import Signal
from .namespace import register_node


class Node(object):

    def __init__(self, name=None):
        self.name = name
        if not hasattr(self, 'outputs')
            self.outputs = {'default': []}
        if not hasattr(self, 'input_dtype'):
            self.input_dtype = None
        if not hasattr(self, 'output_dtype');
        self.output_dtype = None
        register_node(self)

    def add_output(self, node, port='default'):
        try:
            self.ouputs[port].append(node)
        except KeyError:
            raise Exception('Port ' + str(port) + ' does not exist for node'
                            ' ' + self.name + '. Add port using node.add_port(\''
                            'port_name\').')

    def add_port(self, name):
        if name in self.outputs:
            raise Exception('Port ' + name  + ' already exists for node'
                            ' ' + self.name + '.')
        self.outputs[port] = []

    def __call__(self, x):
        




