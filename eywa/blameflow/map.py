from .graph import Node
from .blame import Blame, BlameType

class Map(Node):
    def __init__(self, map, default=None, name=None):
        self.map = map.copy()
        self.rev_map = {v:k for k, v in self.map.items()}
        assert len(self.map) == len(self.rev_map), "Map node requires a 1:1 dictionary."
        self.default = None
        super(Map, self).__init__(name=name)

    def add_input(self, name):
        if self.input_nodes:
            raise Exception("Only single input allowed for Map node.")
        super(Map, self).add_input(name)
        self.input_node = list(self.input_nodes.values())[0]

    def f(self, inputs):
        if self.default is None:
            return self.map[list(inputs.values())[0]]
        return self.map.get(list(inputs.values())[0], self.default)

    def inrange(self, y):
        return y in self.map.values()

    def serialize(self):
        config = super(Map, self).serialize()
        config['map'] = self.map
        config['default'] = self.default
        return config

    def blame(self, blame):
        super(Map, self).blame(blame)
        if blame.blame_type == BlameType.CORRECTIVE:
            self.input_node.blame(blame.fork(expected=self.rev_map[blame.expected]))
        else:
            self.input_node.blame(blame.fork())

    @classmethod
    def deserialize(cls, config):
        if 'class' in config:
            config = config.copy()
            config.pop('class')
        return cls(**config)
