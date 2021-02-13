# backprop

A collection of backpropagation algorithm implementations loosely following Deep Learning, Bengio.

* [Backprop example](./backprop_ex.py): a toy direct recursive implementation treating parameters to be optimized as formal parameters at corresponding nodes rather than leaf nodes. Runs an example optimizing `F[a] = xa^2 + a` with sum and multiply elementary nodes.

* [Algorithms 61 and 62](./algorithms61_62.py) A reverse adjacency list representation of nodes presumed
to satisfy ordering of `v_0, .... v_n-1` as input nodes (leaves)
of graph, and all `v_j in Parent(v_i) => j < i`,
aka all inputs are lesser-indexed. Implements algorithms 61 and 62 following chapter 6 of the Deep Learning Bengio book. 
	*	Example minimizing (non-convex) `f(a,b,c) = a + b + c` with binary add elementary nodes `g(a,b) = a + b`.

	* Example minimizing `f(a, b) = ((a-b)^2 - c^2)^2` with binary nodes `g(a,b) = (a-b)^2`.



