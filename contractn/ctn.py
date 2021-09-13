import networkx as nx

# import opt_einsum as oe

from .nodes import Node
from .edges import Edge
from .utils import assert_valid_tensor, assert_valid_symbol, get_new_symbols


class TN:
    """
    Generic tensor networks which supports copy nodes and weight sharing
    """

    def __init__(self):
        self.G = nx.MultiGraph()
        self._dang_id = 0

    def _init_node(self, node_type, name, edge_symbols, **kwargs):
        """
        Create a new unconnected Node object and add it to the tensor network

        This entails adding several Nodes, one for the Node we actually care
        about, and one dangling node for each edge of the Node we care about.

        Returns an error if used to create a dangling node
        """
        # Check that the name isn't currently used, create node in NX
        name = self._new_node_name(name)
        assert node_type != "dangler"
        self.G.add_node(name)

        # Create the Node object of interest
        node = Node(self, node_type, name, edge_symbols, **kwargs)

        # Create the dangling nodes
        assert len(edge_symbols) == len(node.shape)
        for s, d in zip(edge_symbols, node.shape):
            self._new_dangler(node, d, s)
        assert node.ndim == len(node.shape)

        return node

    def _new_dangler(self, parent, dim, edge_symbol):
        """
        Add a dangler node connected to a non-dangler parent node
        """
        # Add the node to NX
        nx_id = self._dang_id
        assert nx_id not in self.G
        assert parent.name in self.G
        self.G.add_node(nx_id)
        self._dang_id += 1

        # Create Node object, which is added to NX dangler node
        node = Node(self, "dangler", nx_id, (edge_symbol,))

        # Create an edge between dangler and parent node
        self._init_edge(parent, node, dim, edge_symbol)

    def _init_edge(self, node1, node2, dim, edge_symbol):
        """
        Add an edge between two existing Nodes
        """
        # Create the networkx edge
        node1, node2 = node1.name, node2.name
        assert node1 in self.G
        assert node2 in self.G
        key = self.G.add_edge(node1, node2)

        # Create the Edge object
        edge_id = (node1, node2, key)
        Edge(self, edge_id, dim, edge_symbol)

    def add_dense_node(self, tensor, name=None, edge_symbols=None):
        """
        Add a single dense node to the tensor network
        """
        node_type = "dense"
        assert_valid_tensor(tensor)
        edge_symbols = self._new_edge_symbols(
            node_type, tensor.ndim, edge_symbols=edge_symbols
        )
        return self._init_node(node_type, name, edge_symbols, tensor=tensor)

    def add_duplicate_node(self, base_node, name=None, edge_symbols=None):
        """
        Add a single duplicate (clone) node to the tensor network
        """
        node_type = "clone"
        if not isinstance(base_node, Node):
            assert base_node in self.G
            base_node = self.G.nodes[base_node]["tn_node"]
        edge_symbols = self._new_edge_symbols(
            node_type, base_node.ndim, edge_symbols=edge_symbols
        )
        return self._init_node(node_type, name, edge_symbols, base_node=base_node)

    def add_hyperedge_node(self, degree, dimension=None, name=None, edge_symbols=None):
        """
        Add a single hyperedge (copy) node to the tensor network
        """
        node_type = "hyper"
        # Edge symbol lists for hyperedge nodes contain single symbol
        if isinstance(edge_symbols, str):
            edge_symbols = [edge_symbols] * degree
        edge_symbols = self._new_edge_symbols(
            node_type, degree, edge_symbols=edge_symbols
        )
        return self._init_node(
            node_type, name, edge_symbols, degree=degree, dim=dimension
        )

    def add_input_node(self, shape, var_shape_axes=(), name=None, edge_symbols=None):
        """
        Add a single input node to the tensor network
        """
        node_type = "input"
        edge_symbols = self._new_edge_symbols(
            node_type, len(shape), edge_symbols=edge_symbols
        )
        return self._init_node(
            node_type, name, edge_symbols, shape=shape, var_axes=var_shape_axes
        )

    def connect_nodes(self):
        pass

    def _new_node_name(self, name=None):
        """
        Create unused name for node, or check that proposed name is unused
        """
        if name is None:
            name = f"node_{self.num_cores}"
        assert isinstance(name, str)
        if self.G.has_node(name):
            raise TypeError(f"Node name '{name}' already in use")
        return name

    def _new_edge_symbols(self, node_type, degree, edge_symbols=None):
        """
        Create a tuple of unused edge symbols
        """
        # Verify user-specified edge symbols
        if edge_symbols is not None:
            assert len(edge_symbols) == degree
            for es in edge_symbols:
                assert_valid_symbol(es)
            if not self.edge_symbols.isdisjoint(edge_symbols):
                bad_symbol = self.edge_symbols.intersection(edge_symbols).pop()
                raise TypeError(f"Edge symbol '{bad_symbol}' already in use")
            return edge_symbols

        # Generate new edge symbols
        if degree == 0:
            return tuple()
        assert degree > 0
        assert node_type != "dangler"

        if node_type in ("dense", "clone", "input"):
            num_new = degree
        # Symbols are unique, except for hyperedge nodes
        elif node_type == "hyper":
            num_new = 1
        new_symbols = get_new_symbols(self.edge_symbols, num_new)

        return new_symbols if num_new == degree else new_symbols * degree

    @property
    def num_dense(self):
        """
        Returns the number of dense nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["node_type"] == "dense"])

    @property
    def num_duplicate(self):
        """
        Returns the number of duplicate nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["node_type"] == "clone"])

    @property
    def num_hyperedge(self):
        """
        Returns the number of hyperedge nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["node_type"] == "hyper"])

    @property
    def num_input(self):
        """
        Returns the number of input nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["node_type"] == "input"])

    @property
    def num_cores(self):
        """
        Returns the total number of nodes in the tensor network

        This does not include "dangling" nodes in the count, which are just
        placeholders used to indicate uncontracted edges of the network.
        """
        return len([n for n, d in self.G.nodes.data() if d["node_type"] != "dangler"])

    @property
    def edge_symbols(self):
        """
        Return all the edge symbols currently in use
        """
        symbols = [d["symbol"] for _, _, d in self.G.edges(data=True)]
        return set(symbols)
