from node_function import NodeFunction


class NormSquare(NodeFunction):

    """
    Norm squared of a vector
    (x1, x2, ..) |-> x1^2 + x2^2 + ...
    """

    def __init__(self, name):
        super().__init__(name, n_child_nodes=1)

    def evaluate(self, xi):
        output = []
        for row in xi:
            output_row = []
            for e in row:
                output_row.append(e * e)
            output.append(output_row)
        return output_row

    def derivative(self, xi, child_index):
        # id. * 2xi for each row in batch
        N = len(xi)
        m = len(xi[0])
        out_mat = [[[0] * m] * 1] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][0][j] = 2 * xi[i][j]
        return out_mat

    def param_derivative(self, xi, param):
        N = len(xi)
        return [[0.]] * N


