# import random
# from math import prod
from itertools import combinations

# from string import ascii_lowercase as alph

import pytest
import numpy as np
from hypothesis import given, strategies as st

from ..ctn import TN


@pytest.mark.parametrize("graph_topology", ["path", "complete"])
@given(st.integers(2, 6))
def test_connect_dense_nodes(graph_topology, num_nodes):
    """
    Connect dense nodes together in some topology, verify network parameters
    """
    tn = TN()
    bond_dim = 2
    input_dim = 3

    if graph_topology == "path":
        # Initialize nodes
        node_list = []
        shape = (input_dim, bond_dim, bond_dim)
        for _ in range(num_nodes):
            node_list.append(tn.add_dense_node(np.ones(shape)))

        # Connect nodes together
        for i in range(num_nodes - 1):
            node1, node2 = node_list[i], node_list[(i + 1)]
            tn.connect_nodes(node1, node2, 2, 1)

        assert len(tn.nodes()) == tn.num_cores == tn.num_dense == num_nodes
        assert tn.num_duplicate == tn.num_hyperedge == tn.num_input == 0
        assert len(tn.edge_symbols) == len(tn.edges()) == 2 * num_nodes + 1
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes + 2

        for i, node in enumerate(node_list):
            full_neighbors = set(node.neighbors)
            neighbor_set = set(node_list[i - 1 : i] + node_list[i + 1 : i + 2])
            assert len(full_neighbors - neighbor_set) == 1 + int(
                i in [0, num_nodes - 1]
            )
            assert neighbor_set.issubset(full_neighbors)

    elif graph_topology == "complete":
        # Initialize nodes
        node_list = []
        shape = (input_dim,) + (bond_dim,) * (num_nodes - 1)
        for _ in range(num_nodes):
            node_list.append(tn.add_dense_node(np.ones(shape)))

        # Connect nodes together
        for i, j in combinations(range(num_nodes), 2):
            tn.connect_nodes(node_list[i], node_list[j], j, i + 1)

        assert len(tn.nodes()) == tn.num_cores == tn.num_dense == num_nodes
        assert tn.num_duplicate == tn.num_hyperedge == tn.num_input == 0
        assert (
            len(tn.edge_symbols) == len(tn.edges()) == (num_nodes ** 2 + num_nodes) / 2
        )
        assert len(tn.nodes(danglers=True)) == 2 * num_nodes

        for i, node in enumerate(node_list):
            full_neighbors = set(node.neighbors)
            neighbor_set = set(node_list[:i] + node_list[i + 1 :])
            assert len(full_neighbors) == len(neighbor_set) + 1
            assert neighbor_set.issubset(full_neighbors)
