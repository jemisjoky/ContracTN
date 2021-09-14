import random
from math import prod
from string import ascii_lowercase as alph

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..ctn import TN


@given(st.integers(0, 3), st.booleans(), st.booleans())
def test_add_dense_node(order, use_name, use_edge_symbols):
    """Add dense core tensor and verify things look good"""
    tn = TN()
    tensor = np.ones((2,) * order)
    name = "boring_core" if use_name else None
    edge_symbols = tuple(alph[26 - order :]) if use_edge_symbols else None
    good_symbols = tuple(alph[:order]) if edge_symbols is None else edge_symbols
    node = tn.add_dense_node(tensor, name=name, edge_symbols=edge_symbols)

    assert node.node_type == "dense"
    assert node.name == (name if use_name else "node_0")
    assert tn.num_duplicate == tn.num_hyperedge == tn.num_input == 0
    assert tn.num_dense == tn.num_cores == 1
    assert tn.edge_symbols == set(good_symbols)
    assert all(n.node_type == "dangler" for n in node.neighbors)

    assert node.ndim == order
    assert node.shape == tensor.shape
    assert node.edge_symbols == good_symbols
    assert node.size == node.numel == tensor.size

    # Verify that we get errors for non-defined attributes
    for attr in ["base_node", "degree", "dim", "var_axes"]:
        with pytest.raises(Exception):
            getattr(node, attr)


@given(st.integers(0, 3), st.booleans(), st.booleans(), st.booleans())
def test_add_duplicate_node(order, use_dense_name, use_name, use_edge_symbols):
    """Add dense node, then duplicate node and verify things look good"""
    # Add dense node first
    tn = TN()
    tensor = np.ones((2,) * order)
    dense_node = tn.add_dense_node(tensor)
    dense = dense_node.name if use_dense_name else dense_node

    name = "boring_core" if use_name else None
    dense_symbols = set(alph[:order])
    edge_symbols = tuple(alph[26 - order :]) if use_edge_symbols else None
    good_symbols = (
        tuple(alph[order : 2 * order]) if edge_symbols is None else edge_symbols
    )
    node = tn.add_duplicate_node(dense, name=name, edge_symbols=edge_symbols)

    assert node.node_type == "clone"
    assert node.name == (name if use_name else "node_1")
    assert tn.num_hyperedge == tn.num_input == 0
    assert tn.num_dense == tn.num_duplicate == 1
    assert tn.num_cores == 2
    assert tn.edge_symbols == dense_symbols.union(good_symbols)
    assert all(n.node_type == "dangler" for n in node.neighbors)

    assert node.ndim == order
    assert node.shape == tensor.shape
    assert node.edge_symbols == good_symbols
    assert node.base_node is dense_node
    assert node.size == tensor.size
    assert node.numel == 0

    # Verify that we get errors for non-defined attributes
    for attr in ["tensor", "degree", "dim", "var_axes"]:
        with pytest.raises(Exception):
            getattr(node, attr)


@given(st.integers(0, 3), st.booleans(), st.booleans(), st.booleans(), st.booleans())
def test_add_hyperedge_node(order, use_dim, single_symbol, use_name, use_edge_symbols):
    """Add hyperedge node and verify things look good"""
    tn = TN()
    dim = 5 if use_dim else None
    name = "boring_core" if use_name else None
    edge_symbols = None
    if use_edge_symbols:
        edge_symbols = "z" if single_symbol else ("z",) * order
    good_symbols = (("a" if edge_symbols is None else "z"),) * order
    good_shape = ((dim if use_dim else -1),) * order

    # We're not allowing zero order hyperedges
    if order == 0:
        with pytest.raises(Exception):
            node = tn.add_hyperedge_node(
                order, dimension=dim, name=name, edge_symbols=edge_symbols
            )
        return
    node = tn.add_hyperedge_node(
        order, dimension=dim, name=name, edge_symbols=edge_symbols
    )

    assert node.node_type == "hyper"
    assert node.name == (name if use_name else "node_0")
    assert tn.num_duplicate == tn.num_dense == tn.num_input == 0
    assert tn.num_hyperedge == tn.num_cores == 1
    assert tn.edge_symbols == set(good_symbols)
    assert all(n.node_type == "dangler" for n in node.neighbors)

    assert node.ndim == order
    assert node.shape == good_shape
    assert node.edge_symbols == good_symbols
    if use_dim:
        assert node.size == 5 ** order
    else:
        assert node.size is None
    assert node.numel == 0

    # Verify that we get errors for non-defined attributes
    for attr in ["tensor", "base_node", "var_axes"]:
        with pytest.raises(Exception):
            getattr(node, attr)


@given(st.integers(0, 3), st.booleans(), st.booleans(), st.booleans())
def test_add_input_node(order, use_var_axis, use_name, use_edge_symbols):
    """Add input node and verify things look good"""
    tn = TN()
    shape = tuple(random.randint(1, 7) for _ in range(order))
    var_axes = ()
    if use_var_axis and order > 0:
        var_axes = (random.randint(0, order - 1),)
    name = "boring_core" if use_name else None
    edge_symbols = tuple(alph[26 - order :]) if use_edge_symbols else None
    good_symbols = tuple(alph[:order]) if edge_symbols is None else edge_symbols
    node = tn.add_input_node(
        shape, var_shape_axes=var_axes, name=name, edge_symbols=edge_symbols
    )

    assert node.node_type == "input"
    assert node.name == (name if use_name else "node_0")
    assert tn.num_duplicate == tn.num_hyperedge == tn.num_dense == 0
    assert tn.num_input == tn.num_cores == 1
    assert tn.edge_symbols == set(good_symbols)
    assert all(n.node_type == "dangler" for n in node.neighbors)

    assert node.ndim == order
    assert node.shape == tuple(-1 if i in var_axes else d for i, d in enumerate(shape))
    assert node.edge_symbols == good_symbols
    if len(var_axes) == 0:
        assert node.size == prod(shape)
    else:
        assert node.size is None
    assert node.numel == 0

    # Verify that we get errors for non-defined attributes
    for attr in ["tensor", "base_node", "degree", "dim"]:
        with pytest.raises(Exception):
            getattr(node, attr)
