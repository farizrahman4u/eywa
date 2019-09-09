import pytest
from eywa.blameflow import Node, Graph, Select


def test_select_basic():
    a = Node()
    b = Node()
    c = Node()
    s = Select()
    s.add_input('x')
    s.add_input('y')
    assert s({'x': 1,'y': 2, 'selector': 'x'}) == 1
    a.connect(s, 'x')
    b.connect(s, 'y')
    c.connect(s, 'selector')
    a.set_value(10)
    b.set_value(20)
    c.set_value('x')
    assert s.pull() == 10
    a.set_value(30)
    a.push()
    assert s.value == 30
    g = Graph(input_map={'a': a, 'b': b, 'c': c}, final_node=s)
    assert g({'a': 1,'b': 2, 'c': 'x'}) == 1
    g = Graph.deserialize(g.serialize())
    assert g({'a': 1,'b': 2, 'c': 'x'}) == 1

def test_select_partial_exec():
    class Node2(Node):
        def f(self, inputs):
            raise Exception

    a = Node()
    b = Node2()
    c = Node()
    s = Select()
    s.add_input('x')
    s.add_input('y')
    a.connect(s, 'x')
    b.connect(s, 'y')
    c.connect(s, 'selector')
    a.set_value(10)
    b.set_value(20)
    c.set_value('x')
    assert s.pull() == 10

if __name__ == '__main__':
    pytest.main([__file__])
