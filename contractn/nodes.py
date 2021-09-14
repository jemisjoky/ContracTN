from math import prod

from .utils import (
    assert_valid_tensor,
    tensor_attr_error,
    basenode_attr_error,
    degree_attr_error,
    dim_attr_error,
    varaxes_attr_error,
    opposite_node,
)

# Collection of node types which are currently supported
NODE_TYPES = ("dense", "clone", "hyper", "input", "dangler")

# Mandatory and optional information which needs to be given for each node type
NODE_ARGS = {
    "dense": (
        {"tensor"},
        set(),
    ),
    "clone": (
        {"base_node"},
        set(),
    ),
    "hyper": (
        {"degree"},
        {
            "dim",
        },
    ),
    "input": (
        {"shape", "var_axes"},
        set(),
    ),
    "dangler": (set(), set()),
}


class Node:
    """
    Generic node of a TN, which wraps the corresponding node in NetworkX
    """

    def __init__(self, parent_tn, node_type, nx_name, edge_symbols, **kwargs):

        # Keep in mind that most attributes of a Node are derived from the
        # dictionary stored in the associated NX node, so these shouldn't be
        # used until after initialization is completed.

        check_node_args(node_type, kwargs)
        assert nx_name in parent_tn.G
        self.tn = parent_tn
        self.name = nx_name
        self.dict["node_type"] = node_type

        # List of NX edges is needed to order edges (they're unordered in NX)
        n_edges = len(edge_symbols)
        if node_type == "hyper":
            assert len(set(edge_symbols)) == 1
        else:
            assert len(set(edge_symbols)) == n_edges

        if node_type == "dense":
            self.dict["tensor"] = kwargs["tensor"]
            assert n_edges == self.tensor.ndim

        elif node_type == "clone":
            self.dict["base_node"] = kwargs["base_node"]
            if not isinstance(self.base_node, Node):
                self.dict["base_node"] = self.G.nodes[self.base_node]["tn_node"]
            assert self.base_node.node_type == "dense"
            assert n_edges == self.base_node.tensor.ndim

        elif node_type == "hyper":
            self.dict["degree"] = kwargs["degree"]
            assert self.degree > 0, "Hyperedge nodes must have positive degree"
            assert n_edges == self.degree
            self.dict["dim"] = kwargs["dim"] if "dim" in kwargs else None
            assert isinstance(self.dim, int) or self.dim is None

        elif node_type == "input":
            self.dict["_shape"] = kwargs["shape"]
            self.dict["var_axes"] = kwargs["var_axes"]
            assert n_edges == len(self.dict["_shape"])
            assert len(set(self.var_axes)) == len(self.var_axes)
            assert all(0 <= va < n_edges for va in self.var_axes)

        # Make pointer to the Node accessible in networkx node dictionary
        self.dict["tn_node"] = self

        # Create requisite number of dangling nodes, save list of NX edge ids
        edge_names = []
        assert len(self.shape) == len(edge_symbols)
        if node_type != "dangler":
            for i, s in enumerate(edge_symbols):
                edge_names.append(self.tn._new_dangler(self, i, s))
        else:
            # Don't need to create danglers for danglers, just get edge id
            assert len(self.G.edges(self.name)) == 0
            edge_names = list(self.G.edges(self.name, keys=True))
        self._edge_names = edge_names

    @property
    def node_type(self):
        """
        Type of the node
        """
        return self.dict["node_type"]

    @property
    def G(self):
        """
        Networkx graph underlying the node's parent TN
        """
        return self.tn.G

    @property
    def edge_names(self):
        """
        Ordered list of networkx labels for the edges connected to node
        """
        assert set(self._edge_names) == set(self.G.edges(self.name, keys=True))
        return self._edge_names

    @property
    def edges(self):
        """
        Ordered list of edge objects for the modes of the underlying tensor
        """
        return tuple(self.G.edges[en]["tn_edge"] for en in self.edge_names)

    @property
    def edge_symbols(self):
        """
        Ordered list of symbols associated with modes of the underlying tensor
        """
        return tuple(e.symbol for e in self.edges)

    def _dang_name(self, idx):
        """
        Returns name of the dangling node at other end of edge at given index

        Raises an error when the other node isn't a dangler
        """
        # Pull the corresponding node from the edge name, check it's good
        edge_name = self.edge_names[idx]
        dang_name = opposite_node(edge_name, self.name)
        assert len(self.G[self.name][self.dang_name]) == 1
        return dang_name

    @property
    def dict(self):
        """
        Attribute dictionary associated with underlying networkx node
        """
        return self.G.nodes[self.name]

    @property
    def ndim(self):
        """
        Number of edges of the node, i.e. number of modes of the tensor
        """
        ndim = self.G.degree(self.name)
        assert ndim == len(self.shape)
        return ndim

    @property
    def size(self):
        """
        Number of elements in the tensor associated with the Node

        Returns None for Nodes whose underlying tensors don't yet have a
        definite shape. For the literal number of tensor elements stored in
        memory, use Node.numel.
        """
        if self.node_type in ("dense", "clone", "hyper", "input"):
            bad_shape = any(d < 0 for d in self.shape)
            return None if bad_shape else prod(self.shape)

    @property
    def numel(self):
        """
        Number of elements stored in memory for the tensor associated with Node

        Similar to Node.size, but returns 0 for any node types besides dense
        """
        if self.node_type == "dense":
            return prod(self.tensor.shape)
        else:
            return 0

    @property
    def shape(self):
        """
        Shape of the tensor associated with the node

        Values of -1 in the shape tuple indicate an undertermined dimension
        """
        if self.node_type == "dense":
            return self.tensor.shape
        elif self.node_type == "clone":
            return self.base_node.tensor.shape
        elif self.node_type == "hyper":
            return (-1 if self.dim is None else self.dim,) * self.degree
        elif self.node_type == "input":
            return tuple(
                -1 if i in self.var_axes else d
                for i, d in enumerate(self.dict["_shape"])
            )
        elif self.node_type == "dangler":
            return (-1,)

    @property
    def tensor(self):
        """
        Tensor defining a dense node
        """
        if self.node_type != "dense":
            raise tensor_attr_error(self.name, self.node_type)
        return self.dict["tensor"]

    @tensor.setter
    def tensor(self, array):
        if self.node_type != "dense":
            raise tensor_attr_error(self.name, self.node_type)
        assert_valid_tensor(array)
        assert array.ndim == self.ndim
        self.dict["tensor"] = array

    @property
    def base_node(self):
        """
        Base node defining a duplicate node
        """
        if self.node_type != "clone":
            raise basenode_attr_error(self.name, self.node_type)
        return self.dict["base_node"]

    @property
    def degree(self):
        """
        Degree of a hyperedge node
        """
        if self.node_type != "hyper":
            raise degree_attr_error(self.name, self.node_type)
        return self.dict["degree"]

    @property
    def dim(self):
        """
        Dimension of the edges of a hyperedge node
        """
        if self.node_type != "hyper":
            raise dim_attr_error(self.name, self.node_type)
        return self.dict["dim"]

    @property
    def var_axes(self):
        """
        Index numbers of the axes of an input tensor that aren't yet specified
        """
        if self.node_type != "input":
            raise varaxes_attr_error(self.name, self.node_type)
        return self.dict["var_axes"]

    @property
    def neighbors(self):
        """
        Ordered list of nodes which connect to the given node

        Output list is based on the edges of the given node, so that nodes
        which are connected by multiple edges will be listed multiple times.

        For nodes which have dangling edges, a dangling node is placed in the
        appropriate spot.
        """
        return tuple(
            self.G.nodes[opposite_node(e, self.name)]["tn_node"]
            for e in self.edge_names
        )


def check_node_args(node_type, kwdict):
    """
    Ensure input arguments in kwdict are valid for type of node
    """
    assert node_type in NODE_ARGS  # TODO: Replace with actual error message
    mand_args, opt_args = NODE_ARGS[node_type]
    all_args = mand_args.union(opt_args)
    arg_set = set(kwdict.keys())
    if not arg_set.issuperset(mand_args):
        bad_arg = mand_args.difference(arg_set).pop()
        raise TypeError(
            f"Argument '{bad_arg}' missing, needed for node_type '{node_type}'"
        )
    if not all_args.issuperset(arg_set):
        bad_arg = arg_set.difference(all_args).pop()
        raise TypeError(
            f"Argument '{bad_arg}' not recognized for node_type '{node_type}'"
        )
