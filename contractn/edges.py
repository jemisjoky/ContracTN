from .utils import assert_valid_symbol


class Edge:
    """
    Generic edge of a TN, which wraps the corresponding edge in NetworkX
    """

    def __init__(self, parent_tn, nx_id, dim, symbol):
        assert isinstance(nx_id, tuple)
        assert len(nx_id) == 3
        assert nx_id in parent_tn.G.edges
        self.tn = parent_tn
        self.G = self.tn.G
        self.name = nx_id

        assert isinstance(dim, int)
        assert_valid_symbol(symbol)
        self.dict["dim"] = dim
        self.dict["symbol"] = symbol

        # Make pointer to the Edge accessible in networkx edge dictionary
        self.dict["tn_edge"] = self

    @property
    def dict(self):
        """
        Attribute dictionary associated with underlying networkx edge
        """
        return self.G.edges[self.name]

    @property
    def nodes(self):
        return tuple(self.G.node[n]["tn_edge"] for n in self.name[:2])

    @property
    def symbol(self):
        return self.dict["symbol"]

    @property
    def dim(self):
        return self.dict["dim"]

    @property
    def var_dim(self):
        return self.dim < 0
