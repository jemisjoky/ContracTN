# ContracTN

A lightweight reference tensor network library, ContracTN defines TNs as `NetworkX` graphs equipped with additional metadata, which are then compiled into einsum-style strings and contracted together using `opt_einsum`. ContracTN has native support for copy tensors (a.k.a. hyperedges, or diagonal tensors), and uses a novel form of stabilized contraction that mitigates the risk of numerical under/overflow.


## Installation Instructions

To use ContracTN, first clone the repository to your computer (e.g. by calling `git clone git@github.com:jemisjoky/ContracTN.git` on the command line). Navigate to the main directory, and install the necessary dependencies via the command `make requirements`. Finally, call `make install` to make ContracTN accessible on your local machine.


## Examples of Usage

The following examples illustrate various aspects of the usage of ContracTN. They are all available in the Jupyter notebook `contractn/notebooks/ctn_examples.ipynb`.


### Using Copy Tensors

```python
import numpy as np
from contractn import TN
tn = TN()

# Add central copy tensor of order 101
copy_node = tn.add_copy_node(101)

# Connect vectors to all but one edge of the copy tensor
for i in range(100):
    vec = np.array([1, 0.99])
    vec_node = tn.add_dense_node(vec)
    # Connect i'th axis of copy_node to 0'th axis of vec_node
    tn.connect_nodes(copy_node, vec_node, i, 0)

print(tn.contract())  # array([1., 0.36603234])
%time tn.contract()   # 6.85 ms
```

### Specifying TNs using Einsum Strings

```python
# Initialize TNs for 3rd-order Tucker and CP decompositions
cp = TN()
tucker = TN()

# Add central "hub" cores
cp_hub = cp.add_copy_node(3)
tucker_hub = tucker.add_dense_node(np.ones((4, 4, 4)))

# Connect each hub to three factor matrices
for i in range(3):
    mat = np.eye(4, 10)
    cp_mat = cp.add_dense_node(mat)
    tucker_mat = tucker.add_dense_node(mat)
    cp.connect_nodes(cp_hub, cp_mat, i, 0)
    tucker.connect_nodes(tucker_hub, tucker_mat, i, 0)

print(cp.einsum_str)      # "ac,ad,ae->cde"
print(tucker.einsum_str)  # "abc,ae,bf,cg->efg"
```


### Avoiding Overflow with Stable Contraction

```python
# Initialize and connect together vector
# to a chain of 1000 3x3 matrices
tn = TN()
prev_node = tn.add_dense_node(np.ones((3,)))
for i in range(1000):
    mat_node = tn.add_dense_node(np.ones((3, 3)))
    tn.connect_nodes(prev_node, mat_node, -1, 0)
    prev_node = mat_node

print(tn.contract())
    # [inf inf inf]
print(tn.contract(split_format=True))
    # (array([1., 1., 1.]), array(1098.61228867))
```

## Additional Details

More information about the design and rationale behind ContracTN can be found in [our paper](ContracTN_QTNML_2021_Workshop.pdf), presented at the 2021 NeurIPS Quantum Tensor Networks for Machine Learning workshop.