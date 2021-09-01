import numpy as np
from hypothesis import given, strategies as st

from ..ctn import TN
from ..utils import put_in_params

@given(st.integers(0, 4), st.booleans(), st.booleans())
def test_add_node(order, use_name, use_index_names):
    """Add dense core tensor and verify things look good"""
    tn = TN()
    tensor = np.ones((2,) * order)
    name = "boring_core" if use_name else None
    index_names = list(range(order)) if use_index_names else None

    name_out = tn.add_node(tensor, name=name, index_names=index_names)

    assert name_out == (name if use_name else "node_0")
    assert not tn.nodes[name_out]["copy"]
    assert tn.num_copy_nodes == 0
    assert tn.num_cores == 1

    # Conditions for generated index names
    out_idx_names = tn.nodes[name_out]["index_names"]
    if use_index_names:
        assert out_idx_names == index_names
    else:
        assert out_idx_names == [f"idx_{i}" for i in range(order)]

    # Check tensor was only tensor added to params list
    assert put_in_params(tensor, tn._params) == 0
    assert tn.nodes[name_out]["tid"] == 0
    assert len(tn._params) == 1

@given(st.integers(0, 2), st.booleans(), st.booleans(), st.booleans())
def test_add_copy_node(order, use_dim, use_name, use_index_names):
    """Add copy tensor node and verify things look good"""
    tn = TN()
    dimension = 2 if use_dim else None
    name = "boring_copy_node" if use_name else None
    index_names = list(range(order)) if use_index_names else None

    name_out = tn.add_copy_node(order, dimension=dimension, name=name, index_names=index_names)

    assert tn.nodes[name_out]["copy"]
    assert tn.nodes[name_out]["symbol"] is None
    assert tn.nodes[name_out]["dimension"] == dimension
    assert name_out == (name if use_name else "copy_node_0")
    assert tn.num_copy_nodes == 1
    assert tn.num_cores == 0

    # Conditions for generated index names
    out_idx_names = tn.nodes[name_out]["index_names"]
    if use_index_names:
        assert out_idx_names == index_names
    else:
        assert out_idx_names == [f"idx_{i}" for i in range(order)]
