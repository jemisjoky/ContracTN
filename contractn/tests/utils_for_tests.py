from ..einsum import get_func


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
    if backend == "torch":
        from torch import tensor, Tensor

        if not isinstance(tensor1, Tensor):
            tensor1 = tensor(tensor1)
        if not isinstance(tensor2, Tensor):
            tensor2 = tensor(tensor2)
    return get_func("allclose", backend)(tensor1, tensor2)
