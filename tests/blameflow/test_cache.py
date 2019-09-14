from eywa import Node, RandomSwitch, Select
import pytest


def test_no_cache():
    a = Node()
    b = Node()
    c = Node()
    sw = RandomSwitch()
    sw.add_option('a')
    sw.add_option('b')
    sw.add_option('c')
    s = Select()
    s.add_input('a')
    s.add_input('b')
    s.add_input('c')
    a.connect(s, 'a')
    b.connect(s, 'b')
    c.connect(s, 'c')
    sw.connect(s, 'selector')
    a.set_value(1)
    b.set_value(2)
    c.set_value(3)
    vals = [s.pull() for _ in range(30)]
    assert set(vals) == set([1, 2, 3])


if __name__ ==  '__main__':
    pytest.main([__file__])
