from .namespace import register_node


class Node(object):
    def __init__(self, name=None):
        self.name = name
        register_node(self)
        self.input_nodes = {}
        self.output_nodes = []  # [(node, input_name), ...]
        self.input_values = {}
        self.value = None
        self._inputs_changed = False
        self.cache = True

    def set_input_value(self, input_name, value):
        assert input_name in self.input_nodes
        self.input_values[input_name] = value
        self.invalidate_inputs()

    def f(self, inputs):
        return inputs.copy()

    def inputs_ready(self):
        for k in self.input_values:
            if self.input_values[k] is None:
                return False
        return True

    def run(self):
        if  not self.cache or not self.input_nodes or self._inputs_changed and self.inputs_ready():
            self.value = self.f(self.input_values)
            self._inputs_changed = False

    def pull(self):
        if self.cache and self.value is not None and not self._inputs_changed:
            return self.value
        for inp_name, inp_node in self.input_nodes.items():
            if inp_node is None:
                if self.input_values[inp_name] is None:
                    raise Exception("Value not provided for input node " + inp_name)
            else:
                self.input_values[inp_name] = inp_node.pull()
        self.run()
        return self.value

    def invalidate_inputs(self):
        self._inputs_changed = True
        for node, _ in self.output_nodes:
            node.invalidate_inputs()

    def set_value(self, value):
        self.value = value
        self.invalidate_inputs()
        self._inputs_changed = False

    def push(self):
        if self.value is None:
            return
        for output_node, input_name in self.output_nodes:
            output_node.set_input_value(input_name, self.value)
            if output_node.inputs_ready():
                output_node.run()
                output_node.push()

    def connect(self, node, input_name):
        assert isinstance(node, Node)
        assert input_name in node.input_nodes
        self.output_nodes.append((node, input_name))
        node.invalidate_inputs()
        node.input_nodes[input_name] = self
        if not self.cache:
            node.cache = False

    def add_input(self, name):
        if name in self.input_nodes:
            raise Exception("Input with name" + name + " already exists!")
        self.input_nodes[name] = None
        self.input_values[name] = None

    def serialize(self):
        return {'name': None if self._auto_name else self.name,
                'class': self.__class__.__name__,
                'inputs': list(self.input_nodes.keys())}

    def __call__(self, inputs):
        for k in inputs:
            self.set_input_value(k, inputs[k])
        self.run()
        return self.value

    def inrange(self, y):
        return False

    def blame(self, blame):
        blame.node = self

    @classmethod
    def deserialize(cls, config):
        config = config.copy()
        inputs = config.pop('inputs')
        if 'class' in config:
            config.pop('class')
        node = cls(**config)
        for inp in inputs:
            node.add_input(inp)
        return node
