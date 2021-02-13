# algorithms.py
"""
 algorithms following chapter 6 of bengio book
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


def test():
    """
         5
        /\
       3  4
     /\   \
    0  1  2
    """
    def f(arr): return sum(arr)
    def pd(arr): return [1, 1]
    n = 6
    x = [0.5, 0.1, 0.3]
    reverse_adj = {
        5: [3, 4],
        3: [0, 1],
        4: [2]
    }
    nodes = [SimpleNode(f, pd, None) for _ in range(n)]
    cgraph = SimpleCGraph(reverse_adj, nodes)
    root_val = alg61(cgraph, x)
    print(f"root node value: {root_val}")

    grad_table = alg62(cgraph, x)
    print(f"grad table: {grad_table}")


test()
