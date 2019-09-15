from .node import Node
from .namespace import get_node_class


class Graph(Node):
    def __init__(self, input_map, final_node, name=None):
        super(Graph, self).__init__(name)
        # setup inputs
        self.input_map = self._resolve_input_map(input_map)  # {graph_input_name: (node, node_input_name), ...}
        for inp_name in self.input_map:
            self.add_input(inp_name)
        self.final_node = final_node
        self._traverse_graph()

    def _resolve_input_map(self, input_map):
        input_map_2 = {}
        for k, v in input_map.items():
            assert isinstance(k, str)
            if isinstance(v, (list, tuple)):
                assert isinstance(v[0], Node)
                assert isinstance(v[1], (str, type(None)))
                input_map_2[k] = v
            elif isinstance(v, Node):
                assert len(v.input_nodes) <= 1
                input_map_2[k] = (v, None)
            else:
                raise TypeError("Unexpected type in input map: " + str(type(v)))
        return input_map_2

    def _can_reach_final_node(self, node):
        if node == self.final_node:
            return True
        nodes = set([node])
        while nodes:
            curr_node = nodes.pop()
            for onode, _ in curr_node.output_nodes:
                if onode == self.final_node:
                    return True
                nodes.add(onode)
        return False

    def _check_disconnected_inputs(self):
        for v in self.input_map.values():
            inp_node = v[0]
            if not self._can_reach_final_node(inp_node):
                raise Exception("Disconnected graph: unable to reach final " +
                                "node " + self.final_node.name + " from input node" +
                                inp_node.name + " in graph " + self.name)

    def _traverse_graph(self):
        self._check_disconnected_inputs()
        nodes = []
        nodes_set = set()
        for _, (node, _) in self.input_map.items():
            if node not in nodes_set:
                nodes.append(node)
                nodes_set.add(node)
        ptr = 0
        external_nodes = set()
        while ptr < len(nodes):
            next_layer = []
            next_layer_set = set()
            for i in range(ptr, len(nodes)):
                node = nodes[i]
                for ononde, _ in node.output_nodes:
                    if ononde in external_nodes:
                        continue
                    if not self._can_reach_final_node(ononde):
                        external_nodes.add(ononde)
                        continue
                    if ononde not in next_layer_set:
                        next_layer.append(ononde)
                        next_layer_set.add(ononde)
            ptr += len(nodes)
            nodes += next_layer
        self.nodes = nodes

    def f(self, inputs):
        for inp_name, val in inputs.items():
            node, node_inp_name = self.input_map[inp_name]
            if node_inp_name is None:
                if len(node.input_nodes) == 1:
                    node.set_input_value(list(node.input_nodes.keys())[0], val)
                else:
                    node.set_value(val)
            else:
                node.set_input_value(node_inp_name, val)
        return self.final_node.pull()

    def serialize(self):
        config = super(Graph, self).serialize()
        config.pop('inputs')  # redundant
        node_index = {node: i for (i, node) in enumerate(self.nodes)}
        node_configs = [node.serialize() for node in self.nodes]
        input_map_config = {}
        for k, v in self.input_map.items():
            input_map_config[k] = (node_index[v[0]], v[1])
        connectome = {i:list() for i in range(len(self.nodes))}  # {node_index1: [(node_index2, input_name), ...]...}
        for node in self.nodes:
            idx = node_index[node]
            for onode, inp_name in node.output_nodes:
                idx2 = node_index[onode]
                connectome[idx].append((idx2, inp_name))
        config['input_map'] = input_map_config
        config['nodes'] = node_configs
        config['connectome'] = connectome
        config['final_node'] = node_index[self.final_node]
        return config

    @classmethod
    def deserialize(cls, config, custom_objects={}):
        node_configs = config['nodes']
        nodes = []
        for c in node_configs:
            class_name = c['class']
            if class_name in custom_objects:
                node_cls = custom_objects[class_name]
            else:
                node_cls = get_node_class(class_name)
            node = node_cls.deserialize(c)
            nodes.append(node)
        connectome = config['connectome']
        for idx1, outputs in connectome.items():
            for idx2, inp_name in outputs:
                node1 = nodes[idx1]
                node2 = nodes[idx2]
                node1.connect(node2, inp_name)
        input_map_config = config['input_map']
        input_map = {k: (nodes[v[0]], v[1]) for k, v in input_map_config.items()}
        final_node = nodes[config['final_node']]
        name = config['name']
        graph = cls(input_map, final_node, name=name)
        return graph

    def visualize(self, display=True, filename=None):
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise Exception('Error importing graphviz: ' + str(e))
        g = Digraph(self.name, filename=filename)
        g.attr('node', shape='circle', color='orange', style='filled')
        g.attr(size='6,6    ')
        for node in self.nodes:
            for onode, inp_name in node.output_nodes:
                g.edge(node.name + ' (' + node.__class__.__name__ + ')',
                       onode.name + ' (' + onode.__class__.__name__ + ')',
                       label=inp_name)
        if display:
            g.view()
        return g

    def blame(self, blame):
        self.final_node.blame(blame)
