import networkx as nx
import opt_einsum as oe

from utils import assert_valid_tensor, find_in_params


class CTN:
    """
    Base class for generic tensor networks
    """

    def __init__(self):
        self.G = nx.Graph()
        self.params = []

    def add_core(self, tensor, name=None, index_names=None):
        """
        Add a single dense core tensor to the tensor network
        """
        assert_valid_tensor(tensor)

        # Make up a node name if none is given
        if name is None:
            name = f"core_{self.num_cores}"
        assert name not in self.G, f"Node name '{name}' already in use in network"

        # Make up index names if none are given
        if index_names is None:
            index_names = [f"edge_{i}" for i in range(tensor.ndim)]
        assert hasattr(index_names, "__len__"), "index_names must be sequence of names"
        assert (
            len(index_names) == tensor.ndim
        ), f"{len(index_names)} given in index_names, but tensor has {tensor.ndim} indices"

        # Add tensor to params, return index in where core tensor was added
        tid = put_in_params(tensor, self.params)

        self.G.add_node(name, tid=tid, index_names=index_names, pipeline=None, copy=False)

    def add_copy_node(self, order, dimension=None, name=None, index_names=None, symbol=None):
        """
        Add a single copy tensor node to the tensor network
        """
        # Make up a node name if none is given
        if name is None:
            name = f"copy_{self.num_cores}"
        assert name not in self.G, f"Node name '{name}' already in use in network"

        # Make up index names if none are given
        if index_names is None:
            index_names = [f"edge_{i}" for i in range(order)]
        assert hasattr(index_names, "__len__"), "index_names must be sequence of names"
        assert (
            len(index_names) == order
        ), f"{len(index_names)} given in index_names, but copy node has {order} indices"

        self.G.add_node(name, order=order, dimension=dimension, index_names=index_names, symbol=None, copy=True)

    @property
    def num_cores(self):
        """
        Returns the number of core tensors in the tensor network
        """
        return len([n for n, d in self.G.nodes(data=True) if not d["copy"]])

    @property
    def num_copy_nodes(self):
        """
        Returns the number of core tensor nodes in the tensor network
        """
        return len([n for n, d in self.G.nodes(data=True) if d["copy"]])