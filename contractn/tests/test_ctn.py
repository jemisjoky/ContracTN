# import random
# from math import prod
from itertools import combinations

# from string import ascii_lowercase as alph

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..ctn import TN
from .utils_for_tests import assert_index_inverse


@pytest.mark.parametrize("node_type", ["dense", "hyper"])
@pytest.mark.parametrize("graph_topology", ["path", "complete"])
@given(st.integers(2, 6), st.booleans())
def test_connect_nodes(node_type, graph_topology, num_nodes, neg_indices):
    """
    Connect dense or copy nodes in some topology, verify things look good
    """
    tn = TN()
    bond_dim = 2
    input_dim = 3

    if graph_topology == "path":
        if node_type == "dense":
            shape = (input_dim, bond_dim, bond_dim)
            args, kwargs = (np.ones(shape),), {}
            node_method = tn.add_dense_node
        elif node_type == "hyper":
            args, kwargs = (3,), {"dim": bond_dim}
            node_method = tn.add_copy_node

        # Initialize nodes of the appropriate type
        node_list = []
        for _ in range(num_nodes):
            node_list.append(node_method(*args, **kwargs))

        # Connect nodes together
        for i in range(num_nodes - 1):
            node1, node2 = node_list[i], node_list[(i + 1)]
            indices = (-1, -2) if neg_indices else (2, 1)
            tn.connect_nodes(node1, node2, *indices)

        # Check global config
        assert len(tn.nodes()) == tn.num_cores == num_nodes
        assert tn.num_duplicate == tn.num_input == 0
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes + 2
        assert len(tn.edges()) == 2 * num_nodes + 1
        if node_type == "dense":
            assert tn.num_copy == 0
            assert tn.num_dense == num_nodes
            assert len(tn.edge_symbols) == 2 * num_nodes + 1
        elif node_type == "hyper":
            assert tn.num_dense == 0
            assert tn.num_copy == num_nodes
            assert len(tn.edge_symbols) == 1

        # Check local config of each node
        for i, node in enumerate(node_list):
            full_neighbors = set(node.neighbors)
            neighbor_set = set(node_list[i - 1 : i] + node_list[i + 1 : i + 2])
            assert len(full_neighbors - neighbor_set) == 1 + int(
                i in [0, num_nodes - 1]
            )
            assert neighbor_set.issubset(full_neighbors)

    elif graph_topology == "complete":
        if node_type == "dense":
            shape = (input_dim,) + (bond_dim,) * (num_nodes - 1)
            args, kwargs = (np.ones(shape),), {}
            node_method = tn.add_dense_node
        elif node_type == "hyper":
            args, kwargs = (num_nodes,), {"dim": bond_dim}
            node_method = tn.add_copy_node

        # Initialize nodes
        node_list = []
        for _ in range(num_nodes):
            node_list.append(node_method(*args, **kwargs))

        # Connect nodes together
        for i, j in combinations(range(num_nodes), 2):
            indices = (j - num_nodes, i + 1 - num_nodes) if neg_indices else (j, i + 1)
            tn.connect_nodes(node_list[i], node_list[j], *indices)

        # Check global config
        assert len(tn.nodes()) == tn.num_cores == num_nodes
        assert tn.num_duplicate == tn.num_input == 0
        assert len(tn.edges()) == (num_nodes ** 2 + num_nodes) / 2
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes
        if node_type == "dense":
            assert tn.num_copy == 0
            assert tn.num_dense == num_nodes
            assert len(tn.edge_symbols) == (num_nodes ** 2 + num_nodes) / 2
        elif node_type == "hyper":
            assert tn.num_dense == 0
            assert tn.num_copy == num_nodes
            assert len(tn.edge_symbols) == 1

        # Check local config of each node
        for i, node in enumerate(node_list):
            full_neighbors = set(node.neighbors)
            neighbor_set = set(node_list[:i] + node_list[i + 1 :])
            assert len(full_neighbors) == len(neighbor_set) + 1
            assert neighbor_set.issubset(full_neighbors)

    # Make sure _cleanup_edge_symbols doesn't introduce any errors
    tn._cleanup_edge_symbols()
    assert_index_inverse(tn)


@given(st.integers(2, 6), st.booleans(), st.booleans())
def test_remove_edges(num_nodes, single_edges, use_names):
    """
    Form fully-connected TN then remove everything and verify we just have danglers
    """
    # Make fully-connected TN
    tn = TN()
    node_list = [
        tn.add_dense_node(np.ones((2,) * (num_nodes - 1))) for _ in range(num_nodes)
    ]
    for i, j in combinations(range(num_nodes), 2):
        tn.connect_nodes(node_list[i], node_list[j], j - 1, i)

    # Remove all of the edges
    print(len(tn.edges()))
    if single_edges:
        print("One by one")
        for e in tn.edges():
            tn.remove_edge(e.name if use_names else e)
            print(" ", len(tn.edges()))
    else:
        print("All together")
        tn.remove_edges_from([e.name for e in tn.edges()] if use_names else tn.edges())
        print(" ", len(tn.edges()))

    assert tn.num_dense == tn.num_cores == num_nodes
    assert tn.num_duplicate == tn.num_copy == tn.num_input == 0
    assert len(tn.edges()) == len(tn.edge_symbols) == num_nodes * (num_nodes - 1)
    assert all(e.dangler for e in tn.edges())


def test_copy_node_einstring():
    """
    Initialize 3rd-order CP decomposition and verify its einstring is correct

    Based on an issue that came up when finding code samples for the CTN paper.
    """
    cp = TN()

    # Add central "hub" core
    cp_hub = cp.add_copy_node(3)

    # Connect hub to three factor matrices
    for i in range(3):
        mat = np.eye(4, 10)
        cp_mat = cp.add_dense_node(mat)
        cp.connect_nodes(cp_hub, cp_mat, i, 0)

    # Get einstring and verify it is correctly formatted
    einstr = cp.einsum_str
    inputs, output = einstr.split("->")
    inputs = inputs.split(",")
    first_symbols = [s[0] for s in inputs]
    last_symbols = [s[1] for s in inputs]
    assert len(set(first_symbols)) == 1
    assert len(set(last_symbols)) == 3
    assert output == "".join(last_symbols)
