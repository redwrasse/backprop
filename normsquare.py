from node_function import NodeFunction


class NormSquare(NodeFunction):

    """
    Norm squared of a vector over all samples in batch
    1/N sum_i (x1i^2 + x2i^2 + .. )

    evaluates to a single number
    """

    def __init__(self, name):
        super().__init__(name, n_child_nodes=1)

    def evaluate(self, xi):
        N = len(xi)
        sm = 0.
        for row in xi:
            for e in row:
                sm += e * e
        return sm * 1. / N

    def derivative(self, xi, child_index):
        # todo: fix for scalar output!!!!
        pass
        N = len(xi)
        m = len(xi[0])
        deriv = [0.] * m


        # id. * 2xi for each row in batch
        # N = len(xi)
        # m = len(xi[0])
        # out_mat = [[[0] * m] * 1] * N
        # for i in range(N):
        #     for j in range(m):
        #         out_mat[i][0][j] = 2 * xi[i][j]
        # return out_mat

    def param_derivative(self, xi, param):
        return 0.


