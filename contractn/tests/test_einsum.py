import math
import random
from itertools import combinations

import pytest
from hypothesis import given, settings, strategies as st

from ..ctn import TN
from ..einsum import ones, log
from .utils_for_tests import allclose

# Decorater function to try out all backends
backend_list = ("numpy", "torch", "jax", "tensorflow")
all_backends = pytest.mark.parametrize("backend", backend_list)
sole_backend = pytest.mark.parametrize("backend", backend_list[:1])


# @sole_backend
@all_backends
@given(
    st.integers(1, 10),
    st.integers(1, 6),
    st.integers(2, 6),
    st.integers(1, 5),
    st.booleans(),
)
@settings(deadline=None)
def test_contract_mps(
    backend, max_bond_dim, max_input_dim, num_nodes, seed, split_format
):
    """
    Connect all-one tensors as MPS, verify contraction gives correct result
    """
    # Get random bond and input dimensions
    random.seed(seed)
    idims = [random.randint(1, max_input_dim) for _ in range(num_nodes)]
    bdims = [random.randint(1, max_bond_dim) for _ in range(num_nodes - 1)]

    # Initialize all-ones nodes of the MPS
    tn = TN()
    tn.add_dense_node(ones((idims[0], bdims[0]), backend))
    for i in range(1, num_nodes - 1):
        shape = (idims[i], bdims[i - 1], bdims[i])
        tn.add_dense_node(ones(shape, backend))
    tn.add_dense_node(ones((idims[-1], bdims[-1]), backend))
    assert tn.num_cores == num_nodes

    # Connect nodes together
    node_list = tn.nodes()
    for i in range(num_nodes - 1):
        node1, node2 = node_list[i], node_list[(i + 1)]
        tn.connect_nodes(node1, node2, -1, -2 if i != num_nodes - 2 else -1)

    # Contract the TN and verify that we have the right output shape
    output = tn.contract(split_format=split_format)
    if split_format:
        log_value = log(output[0], backend) + output[1]
    else:
        log_value = log(output, backend)
    assert log_value.shape == tuple(idims)

    # Verify that the (constant) values of output tensor are correct
    correct_logval = sum(math.log(bd) for bd in bdims)
    assert allclose(log_value, correct_logval, backend)

    # if node_type == "dense":
    #     shape = (input_dim,) + (bond_dim,) * (num_nodes - 1)
    #     args, kwargs = (np.ones(shape),), {}
    #     node_method = tn.add_dense_node
    # elif node_type == "hyper":
    #     args, kwargs = (num_nodes,), {"dim": bond_dim}
    #     node_method = tn.add_copy_node

    # # Initialize nodes
    # node_list = []
    # for _ in range(num_nodes):
    #     node_list.append(node_method(*args, **kwargs))

    # # Connect nodes together
    # for i, j in combinations(range(num_nodes), 2):
    #     tn.connect_nodes(node_list[i], node_list[j], j, i + 1)

    # # Check global config
    # assert len(tn.nodes()) == tn.num_cores == num_nodes
    # assert tn.num_duplicate == tn.num_input == 0
    # assert len(tn.edges()) == (num_nodes ** 2 + num_nodes) / 2
    # assert len(tn.nodes(danglers=True)) == 2 * num_nodes
    # if node_type == "dense":
    #     assert tn.num_copy == 0
    #     assert tn.num_dense == num_nodes
    #     assert len(tn.edge_symbols) == (num_nodes ** 2 + num_nodes) / 2
    # elif node_type == "hyper":
    #     assert tn.num_dense == 0
    #     assert tn.num_copy == num_nodes
    #     assert len(tn.edge_symbols) == 1

    # # Check local config of each node
    # for i, node in enumerate(node_list):
    #     full_neighbors = set(node.neighbors)
    #     neighbor_set = set(node_list[:i] + node_list[i + 1 :])
    #     assert len(full_neighbors) == len(neighbor_set) + 1
    #     assert neighbor_set.issubset(full_neighbors)
