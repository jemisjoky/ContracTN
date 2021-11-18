import warnings

from .utils import (
    is_valid_tensor,
    is_valid_symbol,
    tensor_attr_error,
    template_name_attr_error,
    change_template_attr_error,
    dim_attr_error,
    prod,
)


class Node:
    """
    Generic node of a graph, which wraps the corresponding node in NetworkX

    This gets subclassed by both Core and Variable, which respectively handle
    tensor cores and variables (hyperedges) of the TN
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
        Whether the node is a tensor core (``True``) or a variable (``False``)
        """
        return bool(self.dict["bipartite"])

    @property
    def nx_degree(self):
        """
        Number of neighbors of the node
        """
        return self.tn.G.degree[self.name]

    @property
    def nx_neighbors(self):
        """
        List of nodes which connect to the given node. These are variables if
        the node is a core, and cores if the node is a variables.
        """
        neighbors = [self.tn.G.nodes[n]["node"] for n in self.tn.G.adj[self.name]]
        assert all(n.is_core != self.is_core for n in neighbors)
        return tuple(neighbors)

    def __repr__(self):
        return (
            f"'Node(name={self.name}, nx_degree={self.nx_degree}, "
            f"nx_neighbors={self.nx_neighbors}, attr_dict={self.dict})'"
        )

    def __str__(self):
        node_class = "Core" if self.is_core else "Variable"
        return f"{node_class}(name={self.name})"


class Core(Node):
    """
    General tensor core, which can be contracted with other cores

    The core that is returned is unconnected to any other cores, and has a
    degree-1 variable (a "dangler") attached to each mode of the tensor. The
    only exception is for copy tensors, which are attached to a single
    degree-n variable.
    """

    # Used to store the template tensors the user has defined
    # TODO: Replace this with something living in state dict of the TN class
    _template_dict = {}

    def __init__(self, parent_tn, name, core_type, var_list, **kwargs):
        # Check input, setup NX structure and basic attributes
        super().__init__(name, parent_tn, True)
        check_core_args(core_type, kwargs)
        self.dict["core_type"] = core_type
        for s in var_list:
            assert is_valid_symbol(s)
            assert s not in self.G  # Variable symbols must be new
        assert len(set(var_list)) == len(var_list)  # Var symbols must be unique
        self.dict["var_list"] = tuple(var_list)

        # Initialization specific to different core types
        if core_type == "dense":
            self.dict["tensor"] = kwargs.pop("tensor")
            assert is_valid_tensor(self.tensor)
            assert self.tensor.ndim == len(var_list)

        elif core_type == "template":
            self.dict["t_name"] = kwargs.pop("template_name")
            # Initialize tensor for this template if it hasn't been set yet
            if self.dict["t_name"] not in self._template_dict:
                self._template_dict[self.dict["t_name"]] = kwargs.pop("tensor")
                assert is_valid_tensor(self.tensor)
            assert self.tensor.ndim == len(var_list)

        elif core_type == "copy":
            self.dict["degree"] = kwargs.pop("degree")
            self.dict["dim"] = kwargs.pop("dim")
            assert self.dict["degree"] >= 1
            assert isinstance(self.dict["dim"], int)
            assert self.dict["dim"] > 0 or self.dict["dim"] == -1
            assert len(self.dict["var_list"]) == 1  # Only one NX neighbor

        elif core_type == "input":
            self.dict["shape"] = tuple(kwargs.pop("shape"))
            self.dict["shape_guess"] = tuple(kwargs.pop("shape_guess"))
            assert len(self.dict["shape_guess"]) == len(self.dict["shape"])
            assert min(self.dict["shape_guess"]) > 0  # Must be defined shape
            assert all(d > 0 or d == -1 for d in self.dict["shape"])

        # Create copy tensors and connect them to modes of the core
        if core_type != "copy":
            for d, symb in zip(self.shape, var_list):
                Variable(symb, parent_tn, d)
                # Note that variables are named by their defining symbol
                self.G.add_edge(self.name, symb)
        else:
            # Although copy tensors have multiple edges, their NX versions
            # only connect to a single variable
            symb = var_list[0]
            Variable(parent_tn, symb, self.dict["degree"], self.dict["dim"])
            self.G.add_edge(self.name, symb)

    @property
    def tensor(self):
        """
        Tensor defining a dense or template node
        """
        if self.is_dense:
            return self.dict["tensor"]
        elif self.is_template:
            return self._template_dict[self.dict["t_name"]]
        else:
            raise tensor_attr_error(self.name, self.core_type)

    @tensor.setter
    def tensor(self, array):
        assert is_valid_tensor(array)
        assert array.ndim == self.ndim
        if self.is_dense:
            self.dict["tensor"] = array
        elif self.is_template:
            warnings.warn(
                "This changes the tensor associated with *all* cores "
                f"of type '{self.template_name}'. You are encouraged to use "
                "`Core.change_template` for this purpose instead."
            )
            self._template_dict[self.dict["t_name"]] = array
        else:
            raise tensor_attr_error(self.name, self.core_type)

    @property
    def core_type(self):
        """
        Type of the node
        """
        return self.dict["core_type"]

    @property
    def is_dense(self):
        return self.dict["core_type"] == "dense"

    @property
    def is_template(self):
        return self.dict["core_type"] == "template"

    @property
    def is_copy(self):
        return self.dict["core_type"] == "copy"

    @property
    def is_input(self):
        return self.dict["core_type"] == "input"

    @property
    def var_list(self):
        """
        Ordered list of networkx labels for the modes of the tensor core
        """
        # Before returning var_list, check that it matches adjacent variables
        var_neighbors = [n.symbol for n in self.nx_neighbors]
        assert sorted(var_neighbors) == sorted(self.dict["var_list"])
        if self.is_copy:
            return self.dict["var_list"] * self.ndim
        else:
            return self.dict["var_list"]

    @property
    def ndim(self):
        """
        Number of edges of the node, i.e. number of modes of the tensor
        """
        if self.is_copy:
            return self.dict["degree"]
        else:
            nd = self.G.degree(self.name)
            assert nd == len(self.shape)
            return nd

    @property
    def size(self):
        """
        Number of elements in the tensor associated with the Core

        Returns None for Cores whose underlying tensors don't yet have a
        definite shape. For the literal number of tensor elements stored in
        memory, use `Core.numel`.
        """
        shape = self.shape
        bad_shape = any(d < 0 for d in shape)
        return None if bad_shape else prod(shape)

    @property
    def numel(self):
        """
        Number of elements stored in memory for the tensor associated with Core

        Similar to `Core.size`, but returns 0 for any core types not requiring
        the storage of a dense tensor.
        """
        if self.is_dense or self.is_template:
            return prod(self.tensor.shape)
        else:
            return 0

    @property
    def shape(self):
        """
        Shape of the tensor associated with the node

        Values of -1 in the shape tuple indicate an undertermined dimension
        """
        if self.is_dense or self.is_template:
            return self.tensor.shape
        elif self.is_copy:
            return (self.dim,) * self.degree
        elif self.is_input:
            return self.dict["shape"]

    @property
    def template_name(self):
        """
        The name of a template Core's defining template
        """
        if self.is_template:
            return self.dict["t_name"]
        else:
            raise template_name_attr_error(self.name, self.core_type)

    def change_template(self, array):
        assert is_valid_tensor(array)
        assert array.ndim == self.ndim
        if not self.is_template:
            raise change_template_attr_error(self.name, self.core_type)
        self._template_dict[self.dict["t_name"]] = array

    @property
    def dim(self):
        """
        Dimension of the modes of a copy node
        """
        if self.is_copy:
            return self.dict["dim"]
        else:
            raise dim_attr_error(self.name, self.core_type)

    @property
    def degree(self):
        """
        Number of edges of the tensor Core
        """
        if self.is_copy:
            return self.dict["degree"]
        else:
            return self.nx_degree

    @property
    def neighbors(self):
        """
        Return list of all neighboring cores
        """
        if self.is_copy:
            var_node = self.G.nodes[self.var_list[0]]
            return [n for n in var_node.nx_neighbors if n is not self]

        else:
            n_list = []
            for var in self.var_list:
                var_node = self.G.nodes[var]
                if var_node.is_copyable:
                    n_list.append(var_node.adjacent_copy)
                else:
                    adj_cores = var_node.nx_neighbors
                    assert len(adj_cores) in (1, 2)
                    if len(adj_cores) == 2:
                        n_list.append(next(c for c in adj_cores if c is not self))

                assert len(n_list) <= self.degree
                return n_list

    def __getitem__(self, key):
        """
        Given an integer index, this gives the variable symbol for that mode.
        Given a variable symbol, this gives the integer index of the associated mode.
        """
        assert isinstance(key, (int, str))
        if isinstance(key, int):
            return self.dict["var_list"][key]
        else:
            assert is_valid_symbol(key)
            return self.dict["var_list"].index(key)

    def __repr__(self):
        return f"'Core_{self.name}(type={self.core_type}, shape={self.shape})'"


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


class Variable(Node):
    """
    Variable node, which acts as a hyperedge to connect tensor cores together
    """

    def __init__(self, symbol, parent_tn, dim):
        # Setup basic attributes
        super().__init__(symbol, parent_tn, False)
        self.dict["dim"] = dim

    @property
    def symbol(self):
        """
        The symbol defining the variable, which is also its NetworkX name
        """
        return self.name

    @property
    def dim(self):
        """
        The dimension associated with a variable
        """
        my_dim = self.dict["dim"]
        assert my_dim is None or isinstance(my_dim, int)
        return my_dim if my_dim is not None else -1

    @property
    def is_copyable(self):
        """
        Whether the variable is attached to a copy node, allowing copying
        """
        return any(n.is_copy for n in self.nx_neighbors)
