import pytest
from eywa.blameflow import Node, Graph
from eywa.blameflow import Blame, BlameType


def test_blame_1():

    class WeightNode(Node):
        def __init__(self, w, *args, **kwargs):
            super(WeightNode, self).__init__(*args, **kwargs)
            self.w = w
        def inrange(self, y):
            return True
        def blame(self, blame):
            super(WeightNode, self).blame(blame)
            assert blame.blame_type == BlameType.CORRECTIVE
            self.w = blame.expected / list(self.input_values.values())[0]
            self.invalidate_inputs()
            blame.node_updated = True

        def f(self, inputs):
            return self.w * list(inputs.values())[0]

    class AddNode(Node):
        def f(self, inputs):
            s = 0
            for v in inputs.values():
                s += v
            return s

        def blame(self, blame):
            super(AddNode, self).blame(blame)
            assert blame.blame_type == BlameType.CORRECTIVE
            for node in self.input_nodes.values():
                node.blame(blame.fork(expected=blame.expected / len(self.input_nodes)))

    a = Node()
    b = Node()
    w1 = WeightNode(3.0)
    w2 = WeightNode(2.0)
    s = AddNode()

    w1.add_input('x')
    a.connect(w1, 'x')

    w2.add_input('x')
    b.connect(w2, 'x')

    s.add_input('x')
    w1.connect(s, 'x')

    s.add_input('y')
    w2.connect(s, 'y')

    a.set_value(1.0)
    b.set_value(2.0)

    assert s.pull() == 7.0

    s.blame(Blame(BlameType.CORRECTIVE, 10.0))

    assert s.pull() == 10.0
