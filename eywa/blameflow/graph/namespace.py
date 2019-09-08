_namespace = {}
_node_classes = {}

def get_node(name):
    try:
        return _namespace[name]
    except KeyError:
        raise KeyError('Unable to find node with name ' + name)


def _auto_name(node):
    node_class_name = node.__class__.__name__.lower()
    idx = 0
    while (True):
        node_name = node_class_name + '_' + str(idx)
        if node_name in _namespace:
            idx += 1
        else:
            return node_name


def register_node(node):
    if node.name is None:
        node.name = _auto_name(node)
        node._auto_name = True
    else:
        if node.name in _namespace:
            if _namespace[node.name] == node:
                raise Exception('Node already registered: ' + node.name)
            else:
                raise Exception('Another node with the same name (' + node.name + ')'
                                                                                  ' already exists.')
        node._auto_name = False
    _namespace[node.name] = node


def register_node_class(cls):
    from .node import Node
    assert issubclass(cls, Node)
    _node_classes[cls.__name__] = cls


def get_node_class(cls_name):
    cls = _node_classes.get(cls_name)
    if cls is None:
        raise Exception("Unregistered node class: " + cls_name)
    return cls
