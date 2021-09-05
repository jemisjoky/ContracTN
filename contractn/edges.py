class Edge:
    """
    Generic edge of a TN, which wraps the corresponding edge in NetworkX
    """

    def __init__(self, G, nx_id, dim, symbol):
        assert isinstance(nx_id, tuple)
        assert len(nx_id) == 3
        assert nx_id in G.edges
        self.G = G
        self.name = nx_id
        self.dict = G.edges[self.name]
        self.dict["symbol"] = symbol
        self.dict["dim"] = dim

        # Make pointer to the Edge accessible in networkx edge dictionary
        self.dict["tn_edge"] = self

    @property
    def nodes(self):
        return tuple(self.G.node[n]["tn_edge"] for n in self.name[:2])

    @property
    def symbol(self):
        return self.dict["symbol"]

    @property
    def dim(self):
        return self.dict["dim"]
