from .signal import Signal
from .namespace import register_node


class Node(object):

    def __init__(self, name=None, ports=None):
        if ports is None:
            ports = ['default']
        self.name = name
        if not hasattr(self, 'outputs'):
            self.outputs = {p: [] for p in ports}
        if not hasattr(self, 'input_dtype'):
            self.input_dtype = None
        if not hasattr(self, 'output_dtype'):
            self.output_dtype = None
        if not hasattr(self, 'parents'):
            self.parents = set()
        register_node(self)

    @property
    def ports(self):
        return self.outputs.keys()

    def add_output(self, node, port='default'):
        try:
            self.outputs[port].append(node)
            if hasattr(self, '_output_nodes_cache'):
                del self._output_nodes_cache
        except KeyError:
            raise Exception('Port ' + str(port) + ' does not exist for node'
                                                  ' ' + self.name + '. Add port using node.add_port(\''
                                                                    'port_name\').')

    def add_port(self, name):
        if name in self.outputs:
            raise Exception('Port ' + name + ' already exists for node'
                                             ' ' + self.name + '.')
        self.outputs[name] = []
        if hasattr(self, '_output_nodes_cache'):
            del self._output_nodes_cache

    def run(self, input, ports=None):
        if ports is None:
            ports = self.outputs.keys()
        for port_name in ports:
            port = self.outputs[port_name]
            for node in port:
                node(input.copy(self))

    def __call__(self, x):
        # TODO Check dtype compatibility
        self.run(x)

    def get_all_output_nodes(self):
        try:
            return self._output_nodes_cache
        except Exception:
            _output_nodes_cache = []
            for p in self.outputs.values():
                for n in p:
                    _output_nodes_cache.append(n)
            self._output_nodes_cache = _output_nodes_cache
            return _output_nodes_cache
