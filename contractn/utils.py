"""Miscellaneous helper functions"""
import operator
from functools import lru_cache, partial, reduce

import opt_einsum as oe


def prod(numbers):
    """
    Compute product of numbers in list (for compatibility with Python <= 3.7)
    """
    return reduce(operator.mul, numbers, 1)


def is_valid_tensor(tensor):
    """
    Check if a user-specified tensor can be used as core tensor data
    """
    return hasattr(tensor, "ndim") and hasattr(tensor, "shape")


def is_valid_symbol(symbol):
    """
    Check if a user-specified symbol is a valid symbol
    """
    return isinstance(symbol, str) and len(symbol) == 1 and not symbol.isdigit()


# def edge_set_equality(edgeset1, edgeset2):
#     """
#     Returns whether two edge sets are equal
#     """
#     es1, es2 = set(edgeset1), set(edgeset2)
#     union = es1 | es2
#     if len(union) == 0:
#         return True
#     assert set(len(t) for t in union) in ({2}, {3})
#     multiset = len(union.pop()) == 3

#     # Order the node labels before comparing, since all graphs are undirected
#     if multiset:
#         es1 = [tuple(sorted(e[:2])) + e[2:] for e in es1]
#         es2 = [tuple(sorted(e[:2])) + e[2:] for e in es2]
#     else:
#         es1 = [tuple(sorted(e)) for e in es1]
#         es2 = [tuple(sorted(e)) for e in es2]
#     return sorted(es1) == sorted(es2)


def get_new_numbers(old_nums, num_new):
    """
    Find new numbers to use for cores of a TN
    """
    num_old = len(old_nums)
    assert min(old_nums) >= 0
    assert len(set(old_nums)) == num_old
    assert all(isinstance(n, int) for n in old_nums)
    max_num = -1 if num_old == 0 else max(old_nums)
    num_gaps = 1 + max_num - num_old

    # Get new numbers lying in gaps between old numbers first
    if num_gaps == 0:
        new_nums = []
    else:
        new_nums = [i for i in range(max_num) if i not in old_nums][:num_new]
    if len(new_nums) < num_new:
        new_nums.extend(range(max_num + 1, max_num + 1 + (num_new - num_gaps)))
    assert len(new_nums) == num_new

    return tuple(new_nums)


def get_new_symbols(old_symbols, num_new):
    """
    Find new opt_einsum symbols to use for variables of a TN
    """
    old_idxs = [symbol_idx(s) for s in old_symbols]
    new_idxs = get_new_numbers(old_idxs, num_new)
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


tensor_attr_error = partial(node_specific_attr_error, "dense and template", "tensor")
template_name_attr_error = partial(
    node_specific_attr_error, "template", "template_name"
)
change_template_attr_error = partial(
    node_specific_attr_error, "template", "change_template"
)
dim_attr_error = partial(node_specific_attr_error, "copy", "dim")
# degree_attr_error = partial(node_specific_attr_error, "copy", "degree")
# varaxes_attr_error = partial(node_specific_attr_error, "input", "var_axes")


full_node_names = {
    "dense": "dense",
    "clone": "duplicate",
    "hyper": "copy",
    "input": "input",
}
