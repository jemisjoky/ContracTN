# ContracTN Planning Document

## Output of a TNet Object

I'm imagining the central class of interest for ContracTN is TNet, which stores all the connectivity information of the associated TN. I'm somewhat inspired by the TensorNetwork interface here, but want to ensure a couple things here:

* Copy tensors are just syntactic sugar for changing the einsum expression defined by a TNet, and are never represented in a dense format.

* Contraction operations are mostly separate from the TNet itself, and only return a contraction result. They never change the tensors associated to the node objects in the model.

The goal is to have a TNet instance used just to define the TN layout, and then call one of its associated methods to produce some derived object that does the actual computation. Some options for this derived object are:

* A function which takes in the unique parameter tensors and returns the contraction associated to the model. This is useful for more functional frameworks like Jax.

* A torch.nn.Module instance which contains the model parameters, and whose forward function carries out the contraction associated to the model. Note that this will require some information about input it is receiving (actually, so will the above).

## Parameter Sharing

It's important to enable parameter sharing in any flexible TN library, so I should separate a collection of _parameter_ tensors giving the unique parameters of the model from _core_ tensors associated with each of the cores in the TN. The former get fed to optimizers and anything else that will change the model values, while the latter get fed to einsum and anything else that directly acts on the individual cores of the TN.

Note that it would be nice to have each node of the TNet be associated with a parameter tensor via a _pipeline_, which is allowed to apply node-specific preprocessing to the parameter tensor. The motivating example here is when we have doubled-up TNs in BMs, where each tensor is used twice, once in its literal form and once in a complex-conjugated form. However, other types of preprocessing could be useful (something for input cores?), so it's best to just treat this as a general-purpose glue operation that I can tinker with later.

## Attributes of a TNet Object

* Actual topology of the TN, stored using NetworkX primitives.

* List of unique parameter tensors.

* Lots of properties coming from basic NetworkX operations.


## Unresolved Questions

* How are we handling input? One way is to add another type of "input core", but how will this differ from standard cores?

* How exactly are copy tensors handled? I could either treat them as regular cores, but with special logic that handles the conversion to an einsum string, or disregard them and just allow edges to connect to multiple nodes. For NetworkX, the former might make more sense.

* What type of graphical structure are we using to represent the TN? I mean this in the sense of how we interface with NetworkX, since it has different classes (?) for defining directed vs. undirected graphs, multigraphs vs. proper graphs, etc. This is related to the question above about copy tensors.