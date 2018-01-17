from .node import Node
class Graph(Node):

    def __init__(self, input, output, *args, **kwargs):
        self.input = input
        self.output = output
        self._traverse()
        super(Graph, self).__init__(*args, **kwargs)

    def _traverse(self):
        nodes = set()
        stack = [self.input]
        while(stack):
            node = stack.pop()
            nodes.add(node)
            node_outs = node.get_all_output_nodes()
            stack += node_outs
        if self.output not in nodes:
            raise Exception('Disconnected graph. Unable to reach '
            + self.output.name + ' from ' + self.input.name + '.')
        for node in nodes:
            node.parents.add(self)
        self.nodes = nodes

    @property
    def outputs(self):
        return self.output.outputs

    @property
    def input_dtype(self):
        return self.input.input_dtype

    @property
    def output_dtype(self):
        return self.output.output_dtype

    def add_output(self, *args, **kwargs):
        return self.output.add_output(*args, **kwargs)

    def add_port(self, *args, **kwargs):
        return self.output.add_port(*args, **kwargs)

    def get_all_output_nodes(self, *args, **kwargs):
        return self.output.get_all_output_nodes(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.input.run(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.input.__call__(*args, **kwargs)
