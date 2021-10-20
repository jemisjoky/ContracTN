from math import prod

from .edges import Edge
from .utils import (
    assert_valid_tensor,
    assert_valid_symbol,
    tensor_attr_error,
    basenode_attr_error,
    degree_attr_error,
    dim_attr_error,
    varaxes_attr_error,
    edge_set_equality,
)


class Node:
    """
    Generic node of a graph, which wraps the corresponding node in NetworkX

    This gets subclassed by both Core and Hyperedge, which respectively handle
    tensor cores and hyperedges of the TN
    """

    def __init__(self, name, parent_tn, is_core):
        # Basic NX node setup, with pointers to parent_tn and Node instance
        self.name = name
        self.tn = parent_tn
        assert name not in self.tn.G
        self.tn.G.add_node(name, tn=parent_tn, node=self, bipartite=int(is_core))

    @property
    def G(self):
        """
        Networkx graph underlying the node's parent TN
        """
        return self.tn.G

    @property
    def dict(self):
        """
        Attribute dictionary associated to the corresponding NetworkX node
        """
        return self.tn.G.nodes[self.name]

    @property
    def is_core(self):
        """
        Whether the node is a tensor core (``True``) or a hyperedge (``False``)
        """
        return bool(self.dict["bipartite"])

    @property
    def _nx_degree(self):
        """
        Number of neighbors of the node
        """
        return self.tn.G.degree[self.name]

    @property
    def _nx_neighbors(self, names_only=False):
        """
        List of nodes which connect to the given node. These are hyperedges if
        the node is a core, and cores if the node is a hyperedge.
        """
        if names_only:
            return tuple(self.tn.G.adj[self.name])
        else:
            return tuple(self.tn.G.nodes[n]["node"] for n in self.tn.G.adj[self.name])

    def __repr__(self):
        node_class = "Core" if self.is_core else "Hyperedge"
        return (
            f"'{node_class}(name={self.name}, nx_degree={self.degree}, "
            f"nx_neighbors={self.neighbors}, attr_dict={self.dict})'"
        )

    def __str__(self):
        node_class = "Core" if self.is_core else "Hyperedge"
        return f"{node_class}(name={self.name})"


class Core(Node):
    """
    General tensor core, which can be contracted with other cores

    The core that is returned is unconnected to any other cores, and has a
    degree-1 hyperedge ("dangler") attached to each mode of the tensor. The
    only exception is for copy tensors, which are attached to a single
    degree-n hyperedge.
    """

    # Used to store the template tensors the user has defined
    _template_dict = {}

    def __init__(self, parent_tn, name, core_type, edge_symbols, **kwargs):
        # Check input, setup NX structure and basic attributes
        super().__init__(name, parent_tn, True)
        check_core_args(core_type, kwargs)
        self.dict["core_type"] = core_type
        for s in edge_symbols:
            assert_valid_symbol(s)
        assert len(set(edge_symbols)) == len(edge_symbols)
        self.dict["edge_symbols"] = tuple(edge_symbols)

        # Initialization specific to different core types
        if core_type == "dense":
            self.dict["tensor"] = kwargs.pop("tensor")
            assert_valid_tensor(self.tensor)
            assert self.tensor.ndim == len(edge_symbols)
            shape = self.tensor.shape

        elif core_type == "template":
            t_name = kwargs.pop("template_name")
            # Initialize tensor for this template if it hasn't been set yet
            if t_name not in self._template_dict:
                assert "tensor" in kwargs
                self._template_dict[t_name] = kwargs.pop("tensor")
                assert_valid_tensor(self._template_dict[t_name])
            self.dict["tensor"] = self._template_dict[t_name]
            assert self.dict["tensor"].ndim == len(edge_symbols)
            shape = self.tensor.shape

        elif core_type == "copy":
            self.dict["degree"] = kwargs.pop("degree")
            self.dict["dim"] = kwargs.pop("dim")
            assert self.dict["degree"] >= 1
            # All edges of copy tensor share the same edge symbol
            assert len(edge_symbols) == 1

        elif core_type == "input":
            shape = kwargs.pop("shape")
            self.dict["shape"] = shape
            self.dict["shape_guess"] = kwargs.pop("shape_guess")
            assert len(self.dict["shape_guess"]) == len(shape)

        # Create hyperedges and connect them to modes of the core
        if core_type != "copy":
            for d, symb in zip(shape, edge_symbols):
                Hyperedge(parent_tn, symb, 1, d)
                # Note that hyperedges are named by their defining edge symbol
                self.G.add_edge(self.name, symb)
        else:
            # Although copy tensor has multiple edges, its NX version
            # only connects to a single hyperedge
            symb = edge_symbols[0]
            Hyperedge(parent_tn, symb, self.dict["degree"], self.dict["dim"])
            self.G.add_edge(self.name, symb)

    @property
    def core_type(self):
        """
        Type of the node
        """
        return self.dict["core_type"]

    @property
    def is_dense(self):
        return self.core_type == "dense"

    @property
    def is_template(self):
        return self.core_type == "template"

    @property
    def is_copy(self):
        return self.core_type == "copy"

    @property
    def is_input(self):
        return self.core_type == "input"


def check_core_args(core_type, kwdict):
    """
    Ensure input arguments in kwdict are valid for given type of core
    """
    if core_type not in _CORE_ARGS:
        raise ValueError(f"Unknown core type '{core_type}'")
    mand_args, opt_args = _CORE_ARGS[core_type]
    all_args = mand_args.union(opt_args)
    arg_set = set(kwdict.keys())
    if not arg_set.issuperset(mand_args):
        bad_arg = mand_args.difference(arg_set).pop()
        raise TypeError(
            f"Argument '{bad_arg}' missing, needed for core type '{core_type}'"
        )
    if not all_args.issuperset(arg_set):
        bad_arg = arg_set.difference(all_args).pop()
        raise TypeError(
            f"Argument '{bad_arg}' not recognized for core type '{core_type}'"
        )


# Mandatory and optional info for each currently supported core type
_CORE_ARGS = {
    "dense": (
        {"tensor"},
        set(),
    ),
    "template": (
        {"template_name"},
        {"tensor"},
    ),
    "copy": (
        {"degree", "dim"},
        set(),
    ),
    "input": (
        {"shape", "shape_guess"},
        set(),
    ),
}


class Hyperedge(Node):
    """
    General hyperedge, which connects tensor cores together
    """

    def __init__(self, parent_tn, symbol, degree, dim):
        # Setup basic attributes
        super().__init__(symbol, parent_tn, False)
        self.dict["degree"] = degree
        self.dict["dim"] = dim

    @property
    def symbol(self):
        """
        The symbol defining the hyperedge, which is also its NetworkX name
        """
        return self.name

    @property
    def degree(self):
        return self.dict["degree"]

    @property
    def dangler(self):
        return self.dict["degree"] == 1

    @property
    def proper_edge(self):
        return self.dict["degree"] == 2

    @property
    def proper_hyperedge(self):
        return self.dict["degree"] > 2
