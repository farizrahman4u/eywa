import pytest

from eywa.blameflow import Node, Graph
from eywa.blameflow import register_node_class


def test_graph_basic_1():
    class AddNode(Node):
        def f(self, inputs):
            return inputs['x'] + inputs['y']
    c = AddNode()
    c.add_input('x')
    c.add_input('y')
    assert c({'x': 2, 'y': 3}) == 5
    a = Node()
    b = Node()
    a.connect(c, 'x')
    b.connect(c, 'y')
    a.set_value(10)
    b.set_value(20)
    assert c.pull() == 30
    a.set_value(15)
    a.push()
    assert c.value == 35
    assert c({'x': 3, 'y': 4}) == 7


def test_graph_basic_2():
    class AddNode(Node):
        def f(self, inputs):
            return inputs['x'] + inputs['y']

    a = Node()
    b = Node()
    c = AddNode()
    c.add_input('x')
    c.add_input('y')
    a.connect(c, 'x')
    b.connect(c, 'y')
    graph = Graph(
        input_map = {'a': a, 'b': b},
        final_node = c
    )
    assert graph({'a': 2, 'b': 3}) == 5

def test_graph_serde():
    class AddNode(Node):
        def f(self, inputs):
            return inputs['x'] + inputs['y']

    register_node_class(AddNode)
    a = Node()
    b = Node()
    c = AddNode()
    c.add_input('x')
    c.add_input('y')
    a.connect(c, 'x')
    b.connect(c, 'y')
    graph = Graph(
        input_map = {'a': a, 'b': b},
        final_node = c
    )
    config = graph.serialize()
    graph2 = Graph.deserialize(config)
    assert graph2({'a': 2, 'b': 3}) == 5

if __name__ == '__main__':
    pytest.main([__file__])
