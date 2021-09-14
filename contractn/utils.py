"""Miscellaneous helper functions"""
from functools import lru_cache, partial

import opt_einsum as oe


def assert_valid_tensor(tensor):
    """
    Check if a user-specified tensor can be used as core tensor data
    """
    assert hasattr(tensor, "ndim")
    assert hasattr(tensor, "shape")


def assert_valid_symbol(symbol):
    """
    Check if a user-specified symbol is a valid symbol
    """
    assert isinstance(symbol, str)
    assert len(symbol) == 1


def get_new_symbols(old_symbols, num_new):
    """
    Find new opt_einsum symbols to use for edges of a TN
    """
    num_symbols = len(old_symbols)
    assert len(set(old_symbols)) == num_symbols
    old_idxs = {symbol_idx(s) for s in old_symbols}
    max_idx = -1 if num_symbols == 0 else max(old_idxs)
    num_gaps = 1 + max_idx - num_symbols

    # Get new symbols lying in gaps between old symbols first
    if num_gaps == 0:
        new_idxs = []
    else:
        new_idxs = [i for i in range(max_idx) if i not in old_idxs][:num_new]
    if len(new_idxs) < num_new:
        new_idxs.extend(range(max_idx + 1, max_idx + 1 + (num_new - num_gaps)))
    assert len(new_idxs) == num_new

    return tuple(oe.get_symbol(idx) for idx in new_idxs)


@lru_cache(maxsize=None)
def symbol_idx(symbol):
    """
    Get the numerical index associated with a symbol, inverse to oe.get_symbol
    """
    assert isinstance(symbol, str) and len(symbol) == 1

    base_symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if symbol in base_symbols:
        idx = base_symbols.index(symbol)
    else:
        idx = ord(symbol) - 140

    assert oe.get_symbol(idx) == symbol
    assert idx >= 0
    return idx


def node_specific_attr_error(node_type, attr_name, node_name, input_ntype):
    """
    Error message generator for node-specific attributes
    """
    full_name = full_node_names[node_type]
    return ValueError(
        f"Only {full_name} nodes have {attr_name} attributes "
        f"(node '{node_name}' has node type '{input_ntype}')"
    )


def opposite_node(edge_id, node):
    """
    Get the neighbor of a given node along a given edge
    """
    assert node in edge_id[:2]
    node_idx = edge_id[:2].index(node)
    return edge_id[(node_idx + 1) % 2]


tensor_attr_error = partial(node_specific_attr_error, "dense", "tensor")
basenode_attr_error = partial(node_specific_attr_error, "clone", "base_node")
degree_attr_error = partial(node_specific_attr_error, "hyper", "degree")
dim_attr_error = partial(node_specific_attr_error, "hyper", "dim")
varaxes_attr_error = partial(node_specific_attr_error, "input", "var_axes")


full_node_names = {
    "dense": "dense",
    "clone": "duplicate",
    "hyper": "hyperedge",
    "input": "input",
}
