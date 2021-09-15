# import random
# from math import prod
from itertools import combinations

# from string import ascii_lowercase as alph

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..ctn import TN


@pytest.mark.parametrize("node_type", ["dense", "hyper"])
@pytest.mark.parametrize("graph_topology", ["path", "complete"])
@given(st.integers(2, 6))
def test_connect_nodes(node_type, graph_topology, num_nodes):
    """
    Connect dense or hyperedge nodes in some topology, verify things look good
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
            node_method = tn.add_hyperedge_node

        # Initialize nodes of the appropriate type
        node_list = []
        for _ in range(num_nodes):
            node_list.append(node_method(*args, **kwargs))

        # Connect nodes together
        for i in range(num_nodes - 1):
            node1, node2 = node_list[i], node_list[(i + 1)]
            tn.connect_nodes(node1, node2, 2, 1)

        # Check global config
        assert len(tn.nodes()) == tn.num_cores == num_nodes
        assert tn.num_duplicate == tn.num_input == 0
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes + 2
        assert len(tn.edges()) == 2 * num_nodes + 1
        if node_type == "dense":
            assert tn.num_hyperedge == 0
            assert tn.num_dense == num_nodes
            assert len(tn.edge_symbols) == 2 * num_nodes + 1
        elif node_type == "hyper":
            assert tn.num_dense == 0
            assert tn.num_hyperedge == num_nodes
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
            node_method = tn.add_hyperedge_node

        # Initialize nodes
        node_list = []
        for _ in range(num_nodes):
            node_list.append(node_method(*args, **kwargs))

        # Connect nodes together
        for i, j in combinations(range(num_nodes), 2):
            tn.connect_nodes(node_list[i], node_list[j], j, i + 1)

        # Check global config
        assert len(tn.nodes()) == tn.num_cores == num_nodes
        assert tn.num_duplicate == tn.num_input == 0
        assert len(tn.edges()) == (num_nodes ** 2 + num_nodes) / 2
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes
        if node_type == "dense":
            assert tn.num_hyperedge == 0
            assert tn.num_dense == num_nodes
            assert len(tn.edge_symbols) == (num_nodes ** 2 + num_nodes) / 2
        elif node_type == "hyper":
            assert tn.num_dense == 0
            assert tn.num_hyperedge == num_nodes
            assert len(tn.edge_symbols) == 1

        # Check local config of each node
        for i, node in enumerate(node_list):
            full_neighbors = set(node.neighbors)
            neighbor_set = set(node_list[:i] + node_list[i + 1 :])
            assert len(full_neighbors) == len(neighbor_set) + 1
            assert neighbor_set.issubset(full_neighbors)

    # Make sure _cleanup_edge_symbols doesn't introduce any errors
    tn._cleanup_edge_symbols()
