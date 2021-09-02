import networkx as nx
import opt_einsum as oe

from .utils import assert_valid_tensor
from .nodes import Node


class TN:
    """
    Generic tensor networks which supports copy nodes and weight sharing
    """

    def __init__(self):
        self.G = nx.Graph()
        self.dict = {"num_dangs": 0}

    def _node_append(self, node_type, name, degree, edge_symbols, **kwargs):
        """
        Create a new unconnected Node object and add it to the tensor network

        This entails adding several Nodes, one for the Node we actually care
        about, and one dangler for each edge of the Node we care about
        """
        # Check that the name isn't currently used, create node in NX
        name = self.new_node_name(name)
        self.G.add_node(name)

        # Create the corresponding Node object
        node = Node(self, node_type, name, edge_symbols, **kwargs)

        return node

    def add_dense_node(self, tensor, edge_symbols=None, name=None):
        """
        Add a single dense node to the tensor network
        """
        pass

    def add_clone_node(self, base_node, edge_symbols=None, name=None):
        """
        Add a single clone (shared) node to the tensor network
        """
        pass

    def add_hyperedge_node(self, order, dimension=None, edge_symbols=None, name=None):
        """
        Add a single hyperedge (copy) node to the tensor network
        """
        pass

    def new_node_name(self, name=None):
        """
        Create new unused name for node, or check that proposed name is unused
        """
        if name is None:
            name = f"node_{self.num_cores}"
        if self.G.has_node(name):
            raise TypeError(f"Node name '{name}' already in use")
        return name

    @property
    def num_dense(self):
        """
        Returns the number of dense nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["type"] == "dense"])

    @property
    def num_clone(self):
        """
        Returns the number of clone nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["type"] == "clone"])

    @property
    def num_hyper(self):
        """
        Returns the number of hyperedge nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["type"] == "hyper"])

    @property
    def num_input(self):
        """
        Returns the number of input nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes.data() if d["type"] == "input"])

    @property
    def num_cores(self):
        """
        Returns the total number of nodes in the tensor network

        This does not include "dangling" nodes in the count, which are just
        placeholders used to indicate uncontracted edges of the network.
        """
        return len([n for n, d in self.G.nodes.data() if d["type"] != "dangler"])

    @property
    def num_dangs(self):
        """
        Returns the number of placeholder dangling nodes in the TN
        """
        return self.dict["num_dangs"]
