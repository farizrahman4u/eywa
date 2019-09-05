from .graph import Node
import random


class Switch(Node):
    def __init__(self, name=None):
        if self.__class__ == Switch:
            raise Exception("Abstract class, do not instantiate!")
        super(Switch, self).__init__(name)
        self.options = set()

    def add_option(self, opt):
        self.options.add(opt)

    def remove_option(self, opt):
        self.options.remove(opt)

    def switch_f(self, inputs):
        raise NotImplementedError("switch_f() not implemented by class " + self.__class__.__name__)

    def f(self, inputs):
        out = self.switch_f(inputs)
        assert out in self.options
        return out

    def inrange(self, y):
        return y in self.options

    def serialize(self):
        config = super(Switch, self).serialize()
        config['options'] = list(self.options)
        return config

    @classmethod
    def deserialize(cls, config):
        sw = cls(name=config['name'])
        for opt in config['options']:
            sw.add_option(opt)
        return sw
    
class RandomSwitch(Switch):
    def __init__(self, name=None):
        super(RandomSwitch, self).__init__(name=name)
        self.cache = False # different output each time

    def switch_f(self, inputs):
        return random.choice(list(self.options))

