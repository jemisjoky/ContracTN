from math import prod

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
        check_node_args(node_type, kwargs)
        self.tn = parent_tn
        self.node_type = node_type
        self.name = nx_name
        self.G = self.tn.G
        self.dict = self.G.nodes[self.name]
        assert nx_name in self.G

        # List of NX edges is needed to order edges (they're unordered in NX)
        self.edges = tuple(edge_symbols)
        n_edges = len(self.edges)

        if node_type == "dense":
            self.tensor = kwargs["tensor"]
            assert n_edges == self.tensor.ndim

        elif node_type == "clone":
            self.base_node = kwargs["base_node"]
            if not isinstance(self.base_node, Node):
                self.base_node = self.G.nodes[self.base_node]["tn_node"]
            assert self.base_node.node_type == "dense"
            assert n_edges == len(self.base_node.tensor.shape)

        elif node_type == "hyper":
            self.degree = kwargs["degree"]
            assert self.degree > 0
            assert n_edges == self.degree
            assert len(set(self.edges)) == 1
            self.dim = kwargs["dim"] if "dim" in kwargs else None

        elif node_type == "input":
            self._shape = kwargs["shape"]
            self.var_axes = kwargs["var_axes"]
            assert n_edges == len(self._shape)

        # Make pointer to the Node accessible in networkx node dictionary
        self.dict["tn_node"] = self
        self.dict["node_type"] = node_type

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
                -1 if i in self.var_axes else d for i, d in enumerate(self._shape)
            )
        elif self.node_type == "dangler":
            # It's simpler to assume danglers don't have a shape
            assert ValueError("Node.shape not supported for dangling nodes")


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
