# algorithms61_62.py
"""
algorithms 61 and 62 following chapter 6 of bengio book

a reverse adjacency list representation of nodes presumed
to satisfy ordering of v_0, .... v_n-1 as input nodes (leaves)
of graph, and all v_j in Parent(v_i) => j < i,
aka all inputs are lesser-indexed.

implements algorithms 61 and 62: forward and backprop on this representation.

"""
import random


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


def run_backprop_algorithm_batches(cgraph, xb, x, param_indices,
                                   data_indices,
                                   batch_size=10,
                                   learning_rate=1e-3,
                                   n_iterations=10**5,
                                   print_freq=10**2):

    ns = len(xb)
    for i in range(n_iterations):
        random.shuffle(xb)
        for j in range(0, ns-batch_size, batch_size):
            xb_minibatch = xb[j:j + batch_size]
            del_ix = {}
            for ix in param_indices:
                del_ix[ix] = 0.

            for xb_e in xb_minibatch:
                for m, xdata_ix in enumerate(data_indices[:-1]):
                    x[xdata_ix] = xb_e[0][m]  # x values
                x[data_indices[-1]] = xb_e[1]  # y value
                alg61(cgraph, x)
                grad_table = alg62(cgraph, x)
                for ix in param_indices:
                    param_update = -learning_rate * grad_table[ix]
                    del_ix[ix] += param_update

            # mini-batch param update
            for ix in param_indices:
                x[ix] += del_ix[ix] / batch_size

        if i % print_freq == 0:
            print(f'param values: {[x[_] for _ in param_indices]}')
            print(f'loss: {cgraph.nodes[-1].val}')


def sample_graph_structure():
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
    return reverse_adj


def mlp_graph_structure(n, k):
    """
    fully connected mlp graph structure
    use convention of first half of the parents of any node the corresponding
    weight parameters,
    the second half the previous layer outputs.

    mlp output is 1d (alg61 and 62 as
    currently implemented support 1d root only)

    k layers each connecting n nodes to n nodes,
    plus final layer as 1 node only.

    So

    nodes 0, ... , n-1 first layer
                0,....,n/2 - 1 is first layer param weights
                n/2,..., n-1 is inputs to first layer(=x)
          n, ..., 2n - 1 second layer, etc.
                n, ..., 3n/2 - 1 is second layer param weights
                3n/2, ..., 2n-1 is inputs to second layer
          n(k-1), ..., nk - 1 second to last (aka kth) layer
                (etc)
          nk node as final layer = output of mlp f(x)

    layer index l = 0, ...., k - 1
            layer l is nodes nl,...., n(l+1) - 1
                            nodes nl, ..., n(l + 1/2) - 1 param weights
                            nodes n(l + 1/2), ... , n(l+1) - 1 is layer inputs


        Then additional two nodes: output target leaf y, and square difference
        (y-f(x))^2.

        nk + 1 node = leaf y
        nk + 2 node = (y - f(x))^2

    """
    if n % 2 != 0:
        raise Exception("n must be even.")
    reverse_adj = {}
    for l in range(k-1):
        weights_indices = list(range(n*l, int(n*(l + 0.5))))
        layer_input_indices = list(range(int(n*(l + 0.5)), n*(l+1)))
        for n1 in weights_indices + layer_input_indices:
            if n1 not in reverse_adj:
                reverse_adj[n1] = []

        next_layer_inputs = list(range(int(n*(l+1 + 0.5)), int(n*((l+2)))))
        for n2 in next_layer_inputs:
            if n2 not in reverse_adj:
                reverse_adj[n2] = []
            for parent_n2 in weights_indices + layer_input_indices:
                reverse_adj[n2].append(parent_n2)

    # output node f(x)
    if n*k not in reverse_adj:
        reverse_adj[n*k] = []
    for parent_n in range(n*(k-1), n*k):
        reverse_adj[n*k].append(parent_n)
    # output target node y
    reverse_adj[n*k + 1] = []
    # node (y - f(x))^2, w/ parents as y, f(x)
    reverse_adj[n*k + 2] = [n*k + 1, n*k]
    return reverse_adj


def example1():

    reverse_adj = sample_graph_structure()

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

    cg1 = cgraph1()
    x = [0.5, 0.1, 0.3]
    print("example 1: min f(a,b,c) = a + b + c")
    run_backprop_algorithm(cg1, x, param_indices=[0, 1, 2])


def example2():
    reverse_adj = sample_graph_structure()

    def cgraph2():
        """
        computational graph of f(a, b) = ((a-b)^2 - c^2)^2
        """

        def f(arr): return (arr[0] - sum(arr[1:])) ** 2

        def pd(arr):
            return [2 * (arr[0] - sum(arr[1:]))] + \
                   [-2 * (arr[0] - sum(arr[1:]))] * (len(arr) - 1)

        n = 6
        nodes = [SimpleNode(f, pd, None) for _ in range(n)]
        cgraph = SimpleCGraph(reverse_adj, nodes)
        return cgraph

    cg2 = cgraph2()
    x = [0.5, 0.1, 0.3]
    # minimizes f(a, b) = ((a-b)^2 - c^2)^2
    print("example 2: min f(a, b) = ((a-b)^2 - c^2)^2")
    run_backprop_algorithm(cg2, x, param_indices=[0, 1])


def mlp_example():
    """
    mlp
    """
    n, k = 6, 2
    reverse_adj = mlp_graph_structure(n, k)

    def cgraph3():

        def f(arr):
            # = <w, x>
            dotp = 0.
            m = int(len(arr)/2)
            for i in range(m):
                dotp += arr[i] * arr[i + m]
            return dotp

        # for (y-f(x))^2
        def fb(arr): return (arr[0] - sum(arr[1:])) ** 2
        def pdb(arr):
            return [2 * (arr[0] - sum(arr[1:]))] + \
                   [-2 * (arr[0] - sum(arr[1:]))] * (len(arr) - 1)

        def pd(arr):
            # y = <w,x>
            # del y del ni = x_i for first half, w_i for second half
            m = int(len(arr)/2)
            return arr[m:] + arr[:m]

        main_nodes = [SimpleNode(f, pd, None) for _ in range(n*k + 1)]
        y_node = [SimpleNode(f, pd, None)]  # dummy node for y (leaf) variable
        sq_loss_node = [SimpleNode(fb, pdb, None)]  # for (y - f(x))^2 node
        nodes = main_nodes + y_node + sq_loss_node
        cgraph = SimpleCGraph(reverse_adj, nodes)
        return cgraph

    cgraph_mlp = cgraph3()
    # all input values plus initial parameter values- aka all leaf values
    # k layers each n / 2 leaf weights,
    # first layer has n total leaves because of additional n/2 inputs
    # plus target y as leaf
    # => (k-1) * n / 2 + n + 1 total # of leaves
    d = int(n/2)  # dim. of input
    x_inputs = [random.normalvariate(0., 1.) for _ in range(d)]
    y_input = 1.
    x = [random.random() for _ in range(int((k-1) * n/2 + n + 1))]
    x[-1] = y_input
    x[d:n] = x_inputs
    # param_indices to train are weights only:
    # nodes nl, ..., n(l + 1/2) - 1 for l = 0,...., k-1
    param_indices = [e for l in range(k) for
                     e in list(range(n*l, int(n*(l + 0.5))))]

    ns = 100  # of training points
    xy_batch = [([random.normalvariate(1., 1.) for dim in range(n)], 1.) for _ \
                in range(ns//2)] + \
               [([random.normalvariate(-1., 1.) for dim in range(n)], -1.) for _ \
                in range(ns // 2)]

    data_indices = list(range(d, n)) + [len(x)-2]  # 2nd to last index = y
    run_backprop_algorithm_batches(cgraph_mlp,
                                   xb=xy_batch,
                                   x=x,
                                   param_indices=param_indices,
                                   data_indices=data_indices)


if __name__ == "__main__":
    # example1()
    # print("----------")
    # example2()
    mlp_example()
