"""Miscellaneous helper functions"""
from functools import lru_cache

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
    old_idxs = [symbol_idx(s) for s in old_symbols]
    min_idx, max_idx = min(old_idxs), max(old_idxs)
    num_gaps = (max_idx - min_idx + 1) - num_symbols

    # Get the new symbols, trying to fill in any gaps in the old symbols
    new_idxs = []
    if num_gaps > 0:
        for idx in range(min_idx + 1, max_idx):
            if idx not in old_idxs:
                new_idxs.append(idx)
    if num_gaps < num_new:
        for idx in range(max_idx + 1, max_idx + (num_new - num_gaps) + 1):
            new_idxs.append(idx)
    new_idxs = new_idxs[:num_new]

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
    return idx
