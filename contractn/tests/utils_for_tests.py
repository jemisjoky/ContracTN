from opt_einsum.backends import get_func

def assert_index_inverse(tn):
    """
    Verify that the Node.__getitem__ and Node.index are inverses
    """
    for node in tn.nodes():
        assert all(node.index(node[i]) == i for i in range(node.ndim))
        assert all(node[node.index(e)].name == e for e in node.edge_names)
        assert all(node[node.index(e)] == e for e in node.edges)

def allclose(tensor1, tensor2, backend):
    """
    Backend-independent check if two tensors are approximately equal
    """
    if backend == "tensorflow":
        backend = "tensorflow.experimental.numpy"
    return get_func("allclose", backend)(tensor1, tensor2)