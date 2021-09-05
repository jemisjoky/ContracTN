class Edge:
    """
    Generic edge of a TN, which wraps the corresponding edge in NetworkX
    """

    def __init__(self, G, nx_id, dim, symbol):
        assert isinstance(nx_id, tuple)
        assert len(nx_id) == 3
        assert nx_id in G.edges
        self.G = G
        self.dim = dim
        self.name = nx_id
        self.symbol = symbol
        self.dict = G.edges[self.name]

        # Get the two Node objects adjacent to the edge
        self.nodes = tuple(G.node[n]["tn_edge"] for n in nx_id[:2])

        # Make pointer to the Edge accessible in networkx edge dictionary
        self.dict["tn_edge"] = self
