import pytest
from eywa.blameflow import Node, Graph
from eywa.blameflow import Switch, RandomSwitch
from eywa.blameflow import Select


def test_switch_1():
    sw = RandomSwitch()
    sw.add_option('x')  # always x
    a = Node()
    b = Node()
    s = Select()
    s.add_input('x')
    s.add_input('y')
    a.connect(s, 'x')
    b.connect(s, 'y')
    sw.connect(s, 'selector')
    a.set_value(10)
    assert s.pull() == 10
