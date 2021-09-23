from functools import partial, reduce

import opt_einsum as oe


def get_einstring(tn):
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


def get_arg_grabber(tn):
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
