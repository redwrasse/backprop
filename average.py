from node_function import NodeFunction


class Average(NodeFunction):
    """ average over all input vector elements
    no parameters to be trained
    """
    def __init__(self, name):
        super().__init__(name, n_child_nodes=1)

    def evaluate(self, xi):
        N = len(xi)
        m = len(xi[0])
        return 1. / (N * m) * sum(a for e in xi for a in e)

    def derivative(self, xi, child_index):
        N = len(xi)
        m = len(xi[0])
        out_mat = [[[0] * m] * 1] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][0][j] = 1. / (N * m)
        return out_mat

    def param_derivative(self, xi, param):
        N = len(xi)
        return [[0.]] * N
