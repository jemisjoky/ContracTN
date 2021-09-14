# ContracTN Planning Document

## Output of a TN Object

I'm imagining the central class of interest for ContracTN is TN, which stores all the connectivity information of the associated TN. I'm somewhat inspired by the TensorNetwork interface here, but want to ensure a couple things here:

* Copy tensors are just syntactic sugar for changing the einsum expression defined by a TN, and are never represented in a dense format.

* Contraction operations are mostly separate from the TN itself, and only return a contraction result. They never change the tensors associated to the node objects in the model.

The goal is to have a TN instance used just to define the TN layout, and then call one of its associated methods to produce some derived object that does the actual computation. Some options for this derived object are:

* A function which takes in the unique parameter tensors and returns the contraction associated to the model. This is useful for more functional frameworks like Jax.

* A torch.nn.Module instance which contains the model parameters, and whose forward function carries out the contraction associated to the model. Note that this will require some information about input it is receiving (actually, so will the above).

## Parameter Sharing

~~It's important to enable parameter sharing in any flexible TN library, so I should separate a collection of _parameter_ tensors giving the unique parameters of the model from _core_ tensors associated with each of the cores in the TN. The former get fed to optimizers and anything else that will change the model values, while the latter get fed to einsum and anything else that directly acts on the individual cores of the TN.~~

~~Note that it would be nice to have each node of the TN be associated with a parameter tensor via a _pipeline_, which is allowed to apply node-specific preprocessing to the parameter tensor. The motivating example here is when we have doubled-up TNs in BMs, where each tensor is used twice, once in its literal form and once in a complex-conjugated form. However, other types of preprocessing could be useful (something for input cores?), so it's best to just treat this as a general-purpose glue operation that I can tinker with later.~~

Nah, here's a better idea. Only have designated data nodes contain tensor attributes, and deal with parameter sharing by having duplicate nodes just point to the original. Allow duplicate nodes to contain other attributes, and to have these attributes take precedence over any attributes of the original node (otherwise, attributes are inherited by default). For any fundamental state changing operations (e.g. in-place changes to the data tensor), make this only allowed on original nodes, and forbidden on duplicate nodes.

The pipeline idea can be implemented by just having a TN method which converts all duplicate tensors into data tensors, and can form part of the last step during a conversion from the unique parameters of a TN to the actual TN that gets contracted by opt_einsum. This will also ensure that only the unique tensors are getting used for the computation, so that we won't get extraneous copies that are stealing gradients from the tensor parameters for the model.


## Graph, Edge, and Node Classes

I've already introduced a TN class built on the NetworkX Graph (or MultiGraph) class, and am tempted to introduce new classes for edges and nodes. Let's see if I can justify this first though. 

Just as the TN class is an overlay of the NetworkX Graph class (supplemented with numerical ingredients), the way that edges and nodes are represented in NetworkX should be accessible via an interface that allows for TN-type considerations. These classes would not only contain graph-based information (connectivity, degree, etc.), but also algebraic information (bond dim, basis, etc.). Custom Node and Edge classes mean I can store that information in a straightforward way, and just create simple wrapper functions in NX to expose the graph-based info.

As bonus points, it could be expeditious to store all of this extra info as a dictionary that we can store in the slot given to us by the NX library. I don't know if this provides any value, but being able to translate a TN instance (minus the methods for the class) into a NX Graph would probably come in handy at times.

Do I need/want to have all of the relevant information for the TN live in that slot all the time? At some times I would like to have it be there, but it doesn't seem strictly necessary to have it _always_ be there. Do keep in mind that NX can do some heavy lifting with providing all nodes and edges which contain certain attributes (using graph.nodes and graph.edges as a function), so for cases where I want to do such a filtered search, it would make sense to store the corresponding attributes as entries in the NX dict for the node/edge.

Oh finally, each NX node/edge should _absolutely_ contain a pointer to the class containing it. This will allow me to use NX lookup functions, get the iterator over NX nodes/edges, and then straightforwardly convert it into an iterator for the custom nodes/edges. Easy peasy.

### Node class attributes

What are some purely linear-algebraic attributes that ctn.TN nodes have which nx.Graph nodes don't?

* `node_type`. This is actually really crucial, as it describes what data to expect from the node, and how that node should be handled in its conversion to whatever output the TN will eventually produce. For example, `copy` would represent one node type, corresponding to copy tensors a.k.a. hyper-edges. This wouldn't contain a tensor attribute, and neither would `duplicate`, which just points to another core for its defining data. See below for a list of node types.

* `name`, which gives the text name of the node. This is distinct from the numerical ID of the NX node, which I'm assuming is just a backend detail, starting at 0 and incrementing upwards. Perhaps it would be easier to use the same ID for each, but for now let's compromise and say that the default ordering for TN nodes is as "name_X", where X is the number of the NX node ID.

* `size`, a property giving the number of elements in the underlying tensor. This is obvious for dense and duplicate nodes, and some thinking is needed for copy and input nodes.

* `order`, the order of the underlying tensor. But this is just the same as the degree of the corresponding edge.

* `norm`, the 2-norm of the tensor.

* `conj`, whether we want to complex conjugate the underlying tensor. This is just one example of general flags we could use to allow last-minute conversions between the data tensor and the actual representation within the network. Nothing is done with them until this conversion stage, and then that preprocessing engine would check for these attributes and handle them accordingly.

* `todo`, a flag which specifies if something needs to be done to prepare the actual value of the core. The representative example here is indexing via an input tensor, where the ideal behavior would be for each indexing operation to be implemented lazily (saves memory), right before it was used in the contraction process.

### Edge class attributes

* `dim`, the dimension of the edge. This is a bond dimension for hidden edges, and an input/output dimension for visible edges. Must be consistent with the underlying tensors in the neighboring nodes.

* `visible`, whether the edge will be contracted over (False), or included as part of a contraction output from the TN (True). This is mostly specified by default, with the rule that all dangling edges (those with a dangler as a neighbor) are visible, and all others are not. The visibility of hidden edges can be set to True, forcing them to be included in the contraction output. Finally, if any hidden edge connected to a copy node is visible, they all should be included as such... although only one of these will eventually appear in the output. Actually, let's just treat this as one of those equivalent structure problems, like rank-1 edges and scalar nodes.

* `symbol`, the symbol which will be assigned to represent the edge when it is converted to einsum. This could probably be left unspecified at first, but whenever we convert the whole thing to an einsum expression, store the associated value in that attribute. Users could maybe play around with naming the indices, but let's try to avoid this scenario as much as possible.

* `set`, whether the bond dim has been determined yet (True), or it is still unspecified (False). This is useful for input nodes, where we don't necessarily have all of the shape data of the to-be-input tensor at hand.

* `target`, a pointer to the intended output node when the TN edges are interpreted as directed.

* There's a lot of intermediate linear-algebraic stuff that gets included in TN gauge fixing and approximation operations, so maybe we could include attributes like that? For example, vectors of Schmidt coefficients could absolutely get included in edges to use for a Vidal-type canonical form for acyclic TNs. Most of these should be handled in the case where we have imposed a directionality on the edges.


## Node types

* `dense`, which contains a dense representation of a tensor as an attribute. 

* `duplicate`, which points to a dense node and just imitates that. Any of the attributes given during the initialization should be interpreted as overriding the attributes of the original dense node, which are otherwise copied without change.

* `copy`, which contains no data and describes a copy tensor. Node that its dimension is left unspecified by default, and is determined when we connect it to something else.

* `input`, which contains no data and serves as an empty spot in the network where we can accept user-specified input cores. These should also allow an indexing operation to be specified, wherein the discrete values of the input tensor are interpreted as one-hot vectors fed to a specified output index.

* `dangler`, which contains no data and serves as a stand-in for the node at the other end of dangling edges.

* Maybe a `tn` node, which contains an entire underlying TN. How feasible would this be to carry out? I'd need some functionality to flatten this, adding all internal nodes and edges into the parent TN, and this would need to be carried out near the end of any conversion into a computable output. This would actually be a pretty cool idea, and the main practical restriction would be internal information to the child networks would be inaccessible to the parent network, and vice versa. Not a big limitation though, and this could really open the door to modularization in TNs.


## Relationship between custom classes and NetworkX objects

Finally implemented all of the custom classes for edges, nodes, and NX graphs, and the layout is really elegant. One pattern I've found very useful is keeping a pointer to the custom class (`Edge`, `Node`, or `TN`) inside the attribute dictionary of the corresponding NX object. This way I can freely transform between custom objects and NX equivalents, but I just realized there's something a bit cooler than this.

I can freely treat a TN object as a NX multigraph, by replacing `self` with `self.G`. Therefore, I can make freely wrap any arbitrary NX algorithm into one acting on TNs, via this simple process:

1. Get NX (multi)graph underlying TN, via `self.G`.
2. Apply NX algorithm to the graph.
3. Recover TN above NX graph, via `self.dict["tn"]`, and return that.

Some of these NX algorithms might not correspond to anything mathematically well-defined in the TN world, but we can at least do something funny and shuffle all the node dictionaries around.

This is powerful but also kinda dangerous, given that it is easy to implement a function which doesn't actually mean anything. For wrapping NX algorithms in this manner, how should I account for transformations of the TN attributes? For starters, some combination of preprocessing and postprocessing of the TNs (additional steps 0 and 4 in above list) could be used to update TN, Node, and Edge attributes based on changes in the underlying graph. I think something more is needed, but that still could be useful.

The contraction process is a canonical example of this relationship between graphical primitives (node merging) and multilinear algebra, expressed here using attributes of the NX graph. More generally, we can see contraction of a TN as just the limit of some graphical process (iterated merging), but translated into a linear algebraic domain. Coding this up is a really nourishing exercise, will be fun to see how this plays out!

## Customizing the Behavior of Contraction

There might be cases where I want certain program logic to be inserted inside of the contraction process. For example, when I'm indexing cores I might not want the core to be indexed until right before it is contracted with.

A sleek way of handling such cases is to run my contraction algorithm with _nodes_ being stored in the tensor queue, rather than literal tensors themselves. At each step of the contraction process, we look at the `todo` attribute of the input nodes, see if there is any processing to do, and if so resolve it to get a backend tensor.

Oooh, we can actually do this with the full contraction process as well! Suppose for example that we want to apply some elementwise activation to the output of our contraction process. Keep a `todo` attribute on the TN class itself, and link it to whatever operation we want to perform. These global todos can be categorized into preprocess and postprocessing for the contraction.

Note though that JAX JIT won't like this stuff, since it will object to the custom classes being fed to the contraction process. Boo, but I'll keep thinking about this.


## Handling the Issues of Rank-1 Edges, Scalars, and Multiple Edges

Edges with a rank of 1 can be considered in different situations to either be not to be edges, and this gives a grey area in how we should encode rank-1 edges within NX.

I think my stance on this is to let rank-1 edges be treated as fully valid edges, but to also provide easily accessible normal form functions, the principal one of which will remove all such edges from the network. When needed, high-use functions which need to be treated as either giving or not giving rank-1 edges in different situations can accept a toggle to give either behavior. For each of these, the default behavior will be to *not* give rank-1 edges.

A similar thing goes for scalars, which can be treated as nodes or not nodes, depending on the circumstances. I'll allow unlimited scalar nodes, but methods should have a heavy bias towards eliminating them.

Also, multiple edges are completely allowed, but make sure we have a normal form function for merging all parallel edges together.


## Unresolved Questions

* How do I ensure that all the lovely behavior I'm adding is JAX-friendly? Practically, keep a running collection of JAX examples in my tests, and check as I develop new features which JAX-specific things break (I'm looking at you, JIT). In terms of overall design, I'm not sure, but handling the customizable behavior in contraction seems like a worthwhile design challenge.

* ~~How are we handling input? One way is to add another type of "input core", but how will this differ from standard cores?~~

* ~~How exactly are copy tensors handled? I could either treat them as regular cores, but with special logic that handles the conversion to an einsum string, or disregard them and just allow edges to connect to multiple nodes. For NetworkX, the former might make more sense.~~

  * ~~What if we just treat copy tensors as standard nodes relative to networkx, but add an attribute saying what type of node we are working with? In this way of viewing TNs, our graphs will always be bipartite between copy nodes and dense (or related) nodes. This can later be packed into an einsum expression pretty easily, where copy tensors give the indices/variables of the einstring and the dense tensors give the terms and operands of the einsum.~~

  * ~~On second thought, let's leave that as optional. I'll have a label indicating whether a node is a copy tensor or a dense tensor (or whatever), but am not going to force the graph to always be bipartite. If needed, I can always add in a function which "biparticizes" an existing graph by (a) adding in order-2 copy tensors on all dense-dense edges, (b) merging any copy tensors which are connected together.~~

  * ~~This biparticized representation could actually be useful for generating the einsum expression, since it is immediate from the two families of nodes (top+bottom, or equivalently variable/core) and the neighbors of each core node.~~

* ~~What type of graphical structure are we using to represent the TN? I mean this in the sense of how we interface with NetworkX, since it has different classes (?) for defining directed vs. undirected graphs, multigraphs vs. proper graphs, etc. This is related to the question above about copy tensors.~~
