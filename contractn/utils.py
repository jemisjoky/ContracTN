"""Miscellaneous helper functions"""


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


def put_in_params(tensor, params):
    """
    Add tensor to parameter list if it isn't already, return index in list
    """
    id_list = [id(t) for t in params]
    if id(tensor) not in id_list:
        params.append(tensor)
        return len(id_list)
    else:
        return id_list.index(id(tensor))
