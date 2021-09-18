from collections import Counter

import networkx as nx

# import opt_einsum as oe

from .nodes import Node
from .edges import Edge
from .utils import assert_valid_tensor, assert_valid_symbol, get_new_symbols


class TN:
    """
    Generic tensor networks which supports copy nodes and weight sharing
    """

    def __init__(self):
        self.G = nx.MultiGraph()
        self._dang_id = 0

    def _init_node(self, node_type, name, edge_symbols, **kwargs):
        """
        Create a new unconnected Node object and add it to the tensor network

        This entails adding several Nodes, one for the Node we actually care
        about, and one dangling node for each edge of the Node we care about.

        Returns an error if used to create a dangling node
        """
        # Check that the name isn't currently used, create node in NX
        name = self._new_node_name(name)
        assert node_type != "dangler"
        self.G.add_node(name)

        # Create the Node object of interest, along with dangling nodes and edges
        node = Node(self, node_type, name, edge_symbols, **kwargs)

        return node

    def _new_dangler(self, parent, idx, edge_symbol):
        """
        Add a dangler node connected to a non-dangler parent node
        """
        # Add the node to NX
        nx_id = f"_dangler_{self._dang_id}"
        assert nx_id not in self.G
        assert parent.name in self.G
        self.G.add_node(nx_id)
        self._dang_id += 1

        # Create Node object, which is added to NX dangler node
        node = Node(self, "dangler", nx_id, (edge_symbol,))

        # Create an edge between dangler and parent node
        return self._init_edge(parent, node, idx, 0, edge_symbol)

    def _init_edge(self, node1, node2, idx1, idx2, edge_symbol):
        """
        Add an edge between two existing Nodes
        """
        # Validate input
        assert isinstance(node1, Node) and isinstance(node2, Node)
        assert_valid_symbol(edge_symbol)
        assert 0 <= idx1 < len(node1.shape)
        assert 0 <= idx2 < len(node2.shape)
        assert node1 in self
        assert node2 in self

        # Check compatibility of (potentially variable-size) mode dimensions
        dim1, dim2 = node1.shape[idx1], node2.shape[idx2]
        var1, var2 = dim1 < 0, dim2 < 0
        if var1 and var2:
            new_dim = -1
        elif var1 != var2:
            new_dim = max(dim1, dim2)
        else:
            assert dim1 == dim2
            new_dim = dim1

        # Create the networkx edge and corresponding Edge object
        n1, n2 = node1.name, node2.name
        edge_id = (n1, n2, self.G.add_edge(n1, n2))
        Edge(self, edge_id, new_dim, edge_symbol)

        # Update the ordered edge list in node
        if node1.node_type != "dangler":
            node1.dict["edge_names"][idx1] = edge_id
        if node2.node_type != "dangler":
            node2.dict["edge_names"][idx2] = edge_id

        # Hyperedge tensors might need some rewriting of edge symbols
        if "hyper" in {node1.node_type, node2.node_type}:
            naughty_node = node1 if node1.node_type == "hyper" else node2
            self._cleanup_edge_symbols(naughty_node)

        return edge_id

    def _remove_edges(self, edge_set):
        """
        Break bonds in the TN, leaving dangling edges
        """
        # Validate input
        all_edges = self.G.edges
        assert all(isinstance(e, Edge) for e in edge_set)
        edge_names = tuple(e.name for e in edge_set)
        assert all(e in all_edges for e in edge_names)

        # For deleted edges, save all affected nodes and associated indices
        aff_nodes, aff_idxs = [], []
        for e in edge_set:
            if e.dangler:
                continue  # Dangling edges can't be removed
            for n in e.name[:2]:
                n = self[n]
                aff_nodes.append(n)
                aff_idxs.append(n.index(e))

        # Remove the edges in networkx graph
        self.G.remove_edges_from(edge_names)

        # Add danglers to replace deleted edges
        new_symbols = get_new_symbols(self.edge_symbols, len(aff_nodes))
        assert len(aff_nodes) == len(aff_idxs) == len(new_symbols)
        for n, idx, es in zip(aff_nodes, aff_idxs, new_symbols):
            n.dict["edge_names"][idx] = self._new_dangler(n, idx, es)

        # Hyperedge tensors might need some rewriting of edge symbols
        if any(n.node_type == "hyper" for n in aff_nodes):
            self._cleanup_edge_symbols()

    def add_dense_node(self, tensor, name=None, edge_symbols=None):
        """
        Add a single dense node to the tensor network
        """
        node_type = "dense"
        assert_valid_tensor(tensor)
        edge_symbols = self._new_edge_symbols(
            node_type, tensor.ndim, edge_symbols=edge_symbols
        )
        return self._init_node(node_type, name, edge_symbols, tensor=tensor)

    def add_duplicate_node(self, base_node, name=None, edge_symbols=None):
        """
        Add a single duplicate (clone) node to the tensor network
        """
        node_type = "clone"
        if not isinstance(base_node, Node):
            assert base_node in self.G
            base_node = self.G.nodes[base_node]["tn_node"]
        edge_symbols = self._new_edge_symbols(
            node_type, base_node.ndim, edge_symbols=edge_symbols
        )
        return self._init_node(node_type, name, edge_symbols, base_node=base_node)

    def add_hyperedge_node(self, degree, dim=None, name=None, edge_symbols=None):
        """
        Add a single hyperedge (copy) node to the tensor network
        """
        node_type = "hyper"
        # Edge symbol lists for hyperedge nodes contain single symbol
        if isinstance(edge_symbols, str):
            edge_symbols = [edge_symbols] * degree
        edge_symbols = self._new_edge_symbols(
            node_type, degree, edge_symbols=edge_symbols
        )
        return self._init_node(node_type, name, edge_symbols, degree=degree, dim=dim)

    def add_input_node(self, shape, var_shape_axes=(), name=None, edge_symbols=None):
        """
        Add a single input node to the tensor network
        """
        node_type = "input"
        edge_symbols = self._new_edge_symbols(
            node_type, len(shape), edge_symbols=edge_symbols
        )
        return self._init_node(
            node_type, name, edge_symbols, shape=shape, var_axes=var_shape_axes
        )

    def connect_nodes(self, node1, node2, index1, index2, edge_symbol=None):
        """
        Create a new edge between two existing nodes along compatible modes
        """
        # Check that edges are dangling
        node1[index1].dangler != node2[index2].dangler

        # Convert node labels to nodes
        if not isinstance(node1, Node):
            assert node1 in self.G
            node1 = self.G[node1]["tn_node"]
        if not isinstance(node2, Node):
            assert node2 in self.G
            node1 = self.G[node2]["tn_node"]

        # Get new edge symbol
        es1, es2 = node1.edge_symbols[index1], node2.edge_symbols[index2]
        if edge_symbol in self.edge_symbols:
            assert edge_symbol in {es1, es2}
        if edge_symbol is None:
            edge_symbol = min(es1, es2)

        # Connect the nodes, remove danglers, update the edge list in nodes
        dang1, dang2 = node1._dang_name(index1), node2._dang_name(index2)
        self.G.remove_node(dang1), self.G.remove_node(dang2)
        self._init_edge(node1, node2, index1, index2, edge_symbol)

    def remove_edge(self, edge):
        """
        Remove a single bond between two non-dangling nodes of the TN
        """
        assert isinstance(edge, (Edge, tuple))
        if isinstance(edge, tuple):
            edge = self.G.edges[edge]["tn_edge"]
        self._remove_edges({edge})

    def remove_edges_from(self, edge_set):
        """
        Remove several bonds between nodes in the TN, leaving dangling edges

        Any edges which are already dangling will be left unchanged
        """
        # Validate input and convert to set of Edge instances
        assert all(isinstance(e, (Edge, tuple)) for e in edge_set)
        edge_set = tuple(edge_set)
        name_edge = [isinstance(e, tuple) for e in edge_set]
        edge_set = set(
            self.G.edges[e]["tn_edge"] if ne else e
            for e, ne in zip(edge_set, name_edge)
        )
        self._remove_edges(edge_set)

    def _new_node_name(self, name=None):
        """
        Create unused name for node, or check that proposed name is unused
        """
        if name is None:
            name = f"node_{self.num_cores}"
        assert isinstance(name, str)
        if self.G.has_node(name):
            raise TypeError(f"Node name '{name}' already in use")
        return name

    def _new_edge_symbols(self, node_type, degree, edge_symbols=None):
        """
        Create a tuple of unused edge symbols
        """
        # Verify user-specified edge symbols
        if edge_symbols is not None:
            assert len(edge_symbols) == degree
            for es in edge_symbols:
                assert_valid_symbol(es)
            if not self.edge_symbols.isdisjoint(edge_symbols):
                bad_symbol = self.edge_symbols.intersection(edge_symbols).pop()
                raise TypeError(f"Edge symbol '{bad_symbol}' already in use")
            return edge_symbols

        # Generate new edge symbols
        if degree == 0:
            return tuple()
        assert degree > 0
        assert node_type != "dangler"

        if node_type in ("dense", "clone", "input"):
            num_new = degree
        # Symbols are unique, except for hyperedge nodes
        elif node_type == "hyper":
            num_new = 1
        new_symbols = get_new_symbols(self.edge_symbols, num_new)

        return new_symbols if num_new == degree else new_symbols * degree

    def _cleanup_edge_symbols(self, naughty_node=None):
        """
        Simplify edge symbols by identifying symbols along connected hyperedges
        """

        # TODO: Ensure that two regions have identical indices *only when*
        #       they are connected by copy tensors. Code below merges indices,
        #       but it doesn't handle the case when edges have been deleted,
        #       creating more disconnected components

        G = self.G
        if naughty_node is None:
            # Get all connected clusters of hyperedge nodes
            hyper_nodes = [n for n, nt in G.nodes(data="node_type") if nt == "hyper"]
            hyper_comps = list(nx.connected_components(G.subgraph(hyper_nodes)))
            symbol_counts = [
                Counter(es for _, _, es in G.edges(comp, data="symbol"))
                for comp in hyper_comps
            ]
            best_symbols = [count.most_common(1)[0][0] for count in symbol_counts]
            for bs, comp in zip(best_symbols, hyper_comps):
                assert_valid_symbol(bs)
                for _, _, d in G.edges.data():
                    d["symbol"] = bs

        else:
            # Get a single cluster of hyperedge nodes
            assert naughty_node.node_type == "hyper"
            hyper_comp = nx.node_connected_component(G, naughty_node.name)
            symbol_count = Counter(
                es for _, _, es in G.edges(hyper_comp, data="symbol")
            )
            best_symbol = symbol_count.most_common(1)[0][0]
            assert_valid_symbol(best_symbol)
            for _, _, d in G.edges.data():
                d["symbol"] = best_symbol

    def nodes(self, as_iter=False, danglers=False):
        """
        Iterator over the Node objects contained in the TN

        Args:
            as_iter: Whether to return nodes as an iterator or a tuple.
                (Default: False)
            danglers: Whether to include dangling nodes, used to
                terminate an unconnected edge in the TN.
                (Default: False)

        Returns:
            node_iter: Iterator or tuple containing all Nodes of the TN, in
                the order they were added.
        """
        node_iter = (d["tn_node"] for n, d in self.G.nodes.data())
        if not danglers:
            node_iter = (n for n in node_iter if n.node_type != "dangler")
        return node_iter if as_iter else tuple(node_iter)

    def edges(self, as_iter=False):
        """
        Iterator over the Edge objects contained in the TN

        Args:
            as_iter: Whether to return edges as an iterator or a tuple.
                (Default: False)
        """
        edge_iter = (e for _, _, e in self.G.edges(data="tn_edge"))
        return edge_iter if as_iter else tuple(edge_iter)

    @property
    def num_dense(self):
        """
        Returns the number of dense nodes in the tensor network
        """
        return len([n for n in self.nodes() if n.node_type == "dense"])

    @property
    def num_duplicate(self):
        """
        Returns the number of duplicate nodes in the tensor network
        """
        return len([n for n in self.nodes() if n.node_type == "clone"])

    @property
    def num_hyperedge(self):
        """
        Returns the number of hyperedge nodes in the tensor network
        """
        return len([n for n in self.nodes() if n.node_type == "hyper"])

    @property
    def num_input(self):
        """
        Returns the number of input nodes in the tensor network
        """
        return len([n for n in self.nodes() if n.node_type == "input"])

    @property
    def num_cores(self):
        """
        Returns the total number of nodes in the tensor network

        This does not include "dangling" nodes in the count, which are just
        placeholders used to indicate uncontracted edges of the network.
        """
        return len(self.nodes())

    @property
    def edge_symbols(self):
        """
        Return all the edge symbols currently in use
        """
        symbols = [d["symbol"] for _, _, d in self.G.edges(data=True)]
        return set(symbols)

    def __contains__(self, node):
        if isinstance(node, Node):
            node = node.name
        return node in self.G

    def __getitem__(self, name):
        """
        Return Node objects based on networkx name
        """
        assert name in self.G
        return self.G.nodes[name]["tn_node"]
