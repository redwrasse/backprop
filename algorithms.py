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
    def __init__(self, f, val=None):
        self.f = f
        self.val = val


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


def test_alg61():
    """
         5
        /\
       3  4
     /\   \
    0  1  2
    """
    def f(arr): return sum(arr)
    n = 6
    x = [0.5, 0.1, 0.3]
    reverse_adj = {
        5: [3, 4],
        3: [0, 1],
        4: [2]
    }
    nodes = [SimpleNode(f, None) for _ in range(n)]
    cgraph = SimpleCGraph(reverse_adj, nodes)
    root_val = alg61(cgraph, x)
    print(root_val)

test_alg61()