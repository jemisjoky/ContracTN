from math import prod

import networkx as nx

from .utils import assert_valid_tensor

# Collection of node types which are currently supported
NODE_TYPES = ("dense", "clone", "hyper", "input", "dangler")

# Mandatory and optional information which needs to be given for each node type
NODE_ARGS = {
    "dense": (
        ("tensor",),
        set(),
    ),
    "clone": (
        ("base_node",),
        set(),
    ),
    "hyper": (
        ("order",),
        {
            "dim",
        },
    ),
    "input": (
        ("shape",),
        set(),
    ),
    "dangler": ((), set()),
}


class Node:
    """
    Generic node of a TN, which wraps the corresponding node in NetworkX
    """

    def __init__(self, G, node_type, nx_name, edges, **kwargs):
        check_node_args(node_type, kwargs)
        assert isinstance(name, str)
        assert nx_name in G
        self.type = node_type
        self.name = nx_name
        self.dict = nx_dict
        # List of NX edges needed to order edges (they're unordered in NX)
        self.edges = edges
        self.G = G
        n_edges = len(edges)

        if node_type == "dense":
            self.tensor = kwargs["tensor"]
            assert n_edges == len(self.tensor.shape)

        elif node_type == "clone":
            self.base = kwargs["base_node"]
            if not isinstance(self.base, Node):
                self.base = self.G.nodes[self.base]["tn_node"]
            assert self.base.type == "dense"
            assert n_edges == len(self.base.tensor.shape)

        elif node_type == "hyper":
            self.order = kwargs["order"]
            self.dim = kwargs["dim"] if "dim" in kwargs else None
            if self.edges is not None:
                assert n_edges == self.order

        elif node_type == "input":
            self.shape = kwargs["shape"]
            assert n_edges == len(self.shape)

        elif node_type == "dangler":
            pass

        # Make pointer to the Node accessible in networkx node dictionary
        self.dict["tn_node"] = self

    @property
    def ndim(self):
        """
        Number of edges of the node, i.e. number of modes of the tensor
        """
        return self.G.degree(self.name)

    @property
    def size(self):
        """
        Number of elements in the tensor associated with the Node

        Returns 0 for any node types besides dense
        """
        if self.type == "dense":
            return prod(self.tensor.shape)
        else:
            return 0


def check_node_args(node_type, kwdict):
    """
    Ensure input arguments in kwdict are valid for type of node
    """
    assert node_type in NODE_ARGS  # TODO: Replace with actual error message
    mand_args, opt_args = NODE_ARGS[node_type]
    arg_set = set(kwdict.keys())
    if not arg_set.issuperset(mand_args):
        bad_arg = set(mand_args).difference(arg_set).pop()
        raise TypeError(
            f"Argument '{bad_arg}' missing, needed for node_type '{node_type}'"
        )
    if not opt_args.issuperset(arg_set):
        bad_arg = arg_set.difference(mand_args.union(opt_args)).pop()
        raise TypeError(
            f"Argument '{bad_arg}' not recognized for node_type '{node_type}'"
        )
