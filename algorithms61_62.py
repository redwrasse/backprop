# algorithms61_62.py
"""
algorithms 61 and 62 following chapter 6 of bengio book

a reverse adjacency list representation of nodes presumed
to satisfy ordering of v_0, .... v_n-1 as input nodes (leaves)
of graph, and all v_j in Parent(v_i) => j < i,
aka all inputs are lesser-indexed.

implements algorithms 61 and 62: forward and backprop on this representation.


"""


class SimpleCGraph:
    # graph as list of nodes and reverse adjacency list
    # presumed to satisfy ordering of v_0, .... v_n-1 as input nodes (leaves)
    # of graph, and all v_j in Parent(v_i) => j < i,
    # aka all inputs are lesser-indexed.
    def __init__(self, reverse_adj, nodes):
        # reverse adj dict: reverse_adj[i] = list of parents of v_i
        # nodes = list of SimpleNodes in order nodes[i] = v_i
        self.reverse_adj = reverse_adj
        self.nodes = nodes


class SimpleNode:
    """ Operates with function f on parent node values
    to hold an evaluated value val """

    def __init__(self, f, pd, val=None):
        self.f = f
        self.pd = pd
        self.val = val

    def partial_derivative(self, child_index, cx):
        return self.pd(cx)[child_index]


def alg61(cgraph, x):
    """ Computes outputs for a given computational graph
    on inputs x, aka forward-propagation
    """
    for i, v in enumerate(x):
        cgraph.nodes[i].val = v
    k = len(x)
    n = len(cgraph.nodes)
    for i in range(k, n):
        parent_indices = cgraph.reverse_adj[i]
        parent_vals = [cgraph.nodes[ix].val for ix in parent_indices]
        cgraph.nodes[i].val = cgraph.nodes[i].f(parent_vals)
    return cgraph.nodes[-1].val


def alg62(cgraph, x):
    """ back-propagation
    grad_table is list of del root / del node
    """
    alg61(cgraph, x)  # forward prop., setting vals at nodes.
    n = len(cgraph.nodes)
    # partial derivatives of all nodes wrt. root
    # grad_table[i] = del root del v_i
    grad_table = [0.] * n
    grad_table[n-1] = 1.
    for j in range(n-2, -1, -1):
        # would be faster with adj list given, not just reverse adj list
        child_indices = []
        grad_table_j = 0.
        for i in cgraph.reverse_adj:
            if j in cgraph.reverse_adj[i]:
                child_indices.append(i)
                parent_vals = [cgraph.nodes[ix].val for ix in cgraph.reverse_adj[i]]
                pd_ix = cgraph.reverse_adj[i].index(j)
                grad_table_j += grad_table[i] * cgraph.nodes[i].partial_derivative(pd_ix, parent_vals)
        grad_table[j] = grad_table_j
    return grad_table


def run_backprop_algorithm(cgraph, x, param_indices,
                           learning_rate=1e-3,
                           n_iterations=10**5,
                           print_freq=10**4):
    """ Run the backpropagation algorithm: alternatively call
    alg61 and alg62.
    Optimize the subset of leaf values identified as parameters by
    the param_indices list.
    """
    for i in range(n_iterations):
        alg61(cgraph, x)
        grad_table = alg62(cgraph, x)
        for ix in param_indices:
            param_update = -learning_rate * grad_table[ix]
            x[ix] += param_update
        if i % print_freq == 0:
            print(f'param values: {[x[_] for _ in param_indices]}')
            print(f'loss: {cgraph.nodes[-1].val}')


def test():
    """
         5
        /\
       3  4
     /\   \
    0  1  2
    """
    reverse_adj = {
            5: [3, 4],
            3: [0, 1],
            4: [2]
        }

    def cgraph1():
        """
        computational graph f(a,b,c) = a + b + c
        """
        def f(arr): return sum(arr)
        def pd(arr): return [1] * len(arr)
        n = 6
        nodes = [SimpleNode(f, pd, None) for _ in range(n)]
        cgraph = SimpleCGraph(reverse_adj, nodes)
        return cgraph

    def cgraph2():
        def f(arr): return (arr[0] - sum(arr[1:]))**2
        def pd(arr):
            return [2 * (arr[0] - sum(arr[1:]))] + \
                   [-2*(arr[0] - sum(arr[1:]))] * (len(arr) - 1)
        n = 6
        nodes = [SimpleNode(f, pd, None) for _ in range(n)]
        cgraph = SimpleCGraph(reverse_adj, nodes)
        return cgraph

    print("example 1: min f(a,b,c) = a + b + c")
    x = [0.5, 0.1, 0.3]
    cg1 = cgraph1()
    # root_val = alg61(cg1, x)
    # print(f"root node value: {root_val}")
    # grad_table = alg62(cg1, x)
    # print(f"grad table: {grad_table}")
    # minimizes f(a,b,c) = a + b + c
    run_backprop_algorithm(cg1, x, param_indices=[0, 1, 2])
    print("----------------")
    print("example 2: min f(a, b) = ((a-b)^2 - c^2)^2")
    cg2 = cgraph2()
    x = [0.5, 0.1, 0.3]
    # minimizes f(a, b) = ((a-b)^2 - c^2)^2
    run_backprop_algorithm(cg2, x, param_indices=[0, 1])


test()
