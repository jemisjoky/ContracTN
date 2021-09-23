import operator
from functools import lru_cache, reduce

import opt_einsum as oe

# Computational functions from opt_einsum
from opt_einsum.backends import get_func
from opt_einsum.contract import parse_backend, _einsum, _tensordot, _transpose


# Some other functions I need for stabilizing the computation
def _zeros(shape, backend):
    """
    Base function for initializing all-zeros tensor
    """
    zeros = get_func("zeros", backend)
    return zeros(shape)


def _where(cond, x, y, backend):
    """
    Base function for conditionally choosing elements from two tensors
    """
    where = get_func("where", backend)
    return where(cond, x, y)


def _abs(x, backend):
    """
    Base function for elementwise absolute value
    """
    abs_func = get_func("abs", backend)
    return abs_func(x)


def _log(x, backend):
    """
    Base function for elementwise logarithm of tensor
    """
    if backend == "tensorflow":
        backend = "tensorflow.math"
    return get_func("log", backend)(x)


def _exp(x, backend):
    """
    Base function for elementwise exponential of tensor
    """
    if backend == "tensorflow":
        backend = "tensorflow.math"
    return get_func("exp", backend)(x)


def _sum(x, backend):
    """
    Base function for summing all elements in a tensor
    """
    func = "sum"
    if backend == "tensorflow":
        backend = "tensorflow.math"
        func = "reduce_sum"
    return get_func(func, backend)(x)


def stabilize(tensor, log_scale, backend):
    """
    Transfer norm from tensor to log_scale register, updating both
    """
    # Minimum norm that is considered non-neglibible
    min_norm = 1e-7

    # Find rescale factor so elements of tensor have average norm of 1
    norm = _sum(_abs(tensor, backend), backend)
    numel = reduce(operator.mul, tensor.shape, 1)
    rescale = norm / numel

    # Rescale only if norm is non-negligible (avoids division by zero)
    rescale_cond = norm > min_norm
    tensor = _where(rescale_cond, tensor / rescale, tensor, backend)
    log_scale = _where(
        rescale_cond, log_scale + _log(rescale_cond, backend), log_scale, backend
    )
    return tensor, log_scale


def destabilize(tensor, log_scale, backend):
    """
    Transfer norm from log_scale register back into tensor
    """
    return tensor * _exp(log_scale, backend)


def make_einstring(tn):
    """
    Convert a TN into an einsum-style string describing its contraction
    """
    # Build einsum terms and free symbols from the nodes of the TN
    ein_terms = []
    free_syms = []
    for node in tn.nodes(as_iter=True, hyperedges=False, danglers=True):
        if node.dangler:
            free_syms.append(node.symbol)
        else:
            ein_terms.append("".join(node.edge_symbols))

    return ",".join(ein_terms) + "->" + "".join(free_syms)


def make_arg_packer(tn):
    """
    Build a function mapping user-given params and input to einsum operands
    """
    # State variables for the scan over TN nodes
    param_list = []  # Position(s) that params appear in operands list
    input_list = []  # Position that each input appears in operands list
    dense_nodes = {}  # Map from dense node names to position in params list
    operand_count = 0  # Running count of how many operands to give to einsum

    # NOTE: Whenever I convert from duplicate to template nodes, change the
    # above to avoid a dense node list, and to add a param_count counter.

    for node in tn.nodes(as_iter=True, hyperedges=False, danglers=False):
        if node.node_type == "dense":
            assert node.name not in dense_nodes
            dense_nodes[node.name] = len(dense_nodes)
            param_list.append([operand_count])

        elif node.node_type == "duplicate":
            base_name = node.base_node.name
            assert base_name in dense_nodes
            param_pos = dense_nodes[base_name]
            assert 0 <= param_pos < len(param_list)
            param_list[param_pos].append(operand_count)

        elif node.node_type == "input":
            input_list.append(operand_count)

        operand_count += 1

    # Verify that all operands are unique and accounted for
    op_inds = [i for idxs in param_list for i in idxs] + input_list
    assert set(op_inds) == set(range(operand_count))
    assert len(set(op_inds)) == len(op_inds)
    np, ni, no = len(param_list), len(input_list), operand_count

    def arg_grabber(params, inputs):
        f"""
        Function packing {np} params and {ni} inputs into {no} einsum operands
        """
        assert len(params) == np
        assert len(inputs) == ni

        operands = [None] * no
        for p, op_list in zip(params, param_list):
            for op_idx in op_list:
                operands[op_idx] = p
        for inp, op_idx in zip(inputs, input_list):
            operands[op_idx] = inp

        assert all(op is not None for op in operands)
        return tuple(operands)

    return arg_grabber


def contract(*operands, **kwargs):
    r"""
    Einsum function with stabilized outputs, based on opt_einsum.contract

    contract(subscripts, *operands, log_format=False, dtype=None, order="K",
    casting="safe", use_blas=True, optimize=True, memory_limit=None, backend="numpy")

    Evaluates the Einstein summation convention on the operands. A drop in
    replacement for NumPy's einsum function that optimizes the order of contraction
    to reduce overall scaling at the cost of several intermediate arrays.

    Args:
        einstr (str): Einsum string describing the contraction layout.
        *operands (list of array_like): Sequence of tensors to be contracted together.
        log_format (bool): Whether to split output into two parts, the first
            giving a rescaled version of the output, and the second giving the
            logarithm of the scaling factor. This avoids underflow and overflow
            of large contraction networks.
        optimize (str): Specifies the type of optimization to use path (see below).

            - if a list is given uses this as the path.
            - `"optimal"` An algorithm that explores all possible ways of
            contracting the listed tensors. Scales factorially with the number of
            terms in the contraction.
            - `"dp"` A faster (but essentially optimal) algorithm that uses
            dynamic programming to exhaustively search all contraction paths
            without outer-products.
            - `"greedy"` An cheap algorithm that heuristically chooses the best
            pairwise contraction at each step. Scales linearly in the number of
            terms in the contraction.
            - `"random-greedy"` Run a randomized version of the greedy algorithm
            32 times and pick the best path.
            - `"random-greedy-128"` Run a randomized version of the greedy
            algorithm 128 times and pick the best path.
            - `"branch-all"` An algorithm like optimal but that restricts itself
            to searching "likely" paths. Still scales factorially.
            - `"branch-2"` An even more restricted version of "branch-all" that
            only searches the best two options at each step. Scales exponentially
            with the number of terms in the contraction.
            - `"auto"` Choose the best of the above algorithms whilst aiming to
            keep the path finding time below 1ms.
            - `"auto-hq"` Aim for a high quality contraction, choosing the best
            of the above algorithms whilst aiming to keep the path finding time
            below 1 sec.

        dtype (str): The dtype of the given contraction, see np.einsum.
        order (str): The order of the resulting contraction, see np.einsum.
        casting (str): The casting procedure for operations of different dtype,
            see np.einsum.
        use_blas (bool): Whether to use BLAS for valid operations, may use extra
            memory for more intermediates.

        memory_limit ({None, int, "max_input"} (default: `None`)): Give the upper
        bound of the largest intermediate tensor contract will build.
            - None or -1 means there is no limit
            - `max_input` means the limit is set as largest input tensor
            - a positive integer is taken as an explicit limit on the number of elements

            The default is None. Note that imposing a limit can make contractions
            exponentially slower to perform.

        backend (str, optional (default: ``auto``)): Which library to use to perform
            the required ``tensordot``, ``transpose`` and ``einsum`` calls. Should
            match the types of arrays supplied, See :func:`contract_expression`
            for generating expressions which convert numpy arrays to and from the
            backend library automatically.

    **Notes:**

    This function should produce a result identical to that of NumPy's einsum
    function. The primary difference is ``contract`` will attempt to form
    intermediates which reduce the overall scaling of the given einsum contraction.
    By default the worst intermediate formed will be equal to that of the largest
    input array. For large einsum expressions with many input arrays this can
    provide arbitrarily large (1000 fold+) speed improvements.

    For contractions with just two tensors this function will attempt to use
    NumPy's built-in BLAS functionality to ensure that the given operation is
    preformed optimally. When NumPy is linked to a threaded BLAS, potential
    speedups are on the order of 20-100 for a six core machine.
    """
    optimize_arg = kwargs.pop("optimize", True)
    if optimize_arg is True:
        optimize_arg = "auto"

    valid_einsum_kwargs = ["dtype", "order", "casting"]
    einsum_kwargs = {k: v for (k, v) in kwargs.items() if k in valid_einsum_kwargs}

    # Grab non-einsum kwargs
    use_blas = kwargs.pop("use_blas", True)
    log_format = kwargs.pop("log_format", False)
    memory_limit = kwargs.pop("memory_limit", None)
    backend = kwargs.pop("backend", "auto")

    # Make sure remaining keywords are valid for einsum
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError(
            "Did not understand the following kwargs: {}".format(unknown_kwargs)
        )

    # Build the contraction list from shapes, which is cached
    assert isinstance(operands[0], str)
    einstr, op_shapes = operands[0], [op.shape for op in operands[1:]]
    contract_list = _contract_path(
        einstr,
        op_shapes,
        optimize=optimize_arg,
        memory_limit=memory_limit,
        use_blas=use_blas,
    )

    result, log_scale = _core_contract(
        operands, contract_list, backend, **einsum_kwargs
    )

    if log_format:
        return result, log_scale
    else:
        return destabilize(result, log_scale)


@lru_cache()
def _contract_path(einstr, operand_shapes, **kwargs):
    """
    Cached version of opt_einsum.contract_path acting on operand shapes

    Only returns the contraction path, and not any info about that path.
    """
    kwargs["shapes"] = True
    kwargs["einsum_call"] = True
    _, contract_list = oe.contract_path(einstr, *operand_shapes, **kwargs)
    return contract_list