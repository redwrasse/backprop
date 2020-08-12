from node_function import NodeFunction


class ConstMult(NodeFunction):
    """
    mult by a constant
    """
    def __init__(self, name, const):
        super().__init__(name, n_child_nodes=1)
        self.const = const

    def evaluate(self, xi):
        output = []
        for row in xi:
            out_row = []
            for e in row:
                out_row.append(e * self.const)
            output.append(out_row)
        return output

    def derivative(self, xi, child_index):
        # identity matrix times const for each row in batch
        N = len(xi)
        m = len(xi[0])
        out_mat = [[[0] * m] * m] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][j][j] = self.const
        return out_mat

    def param_derivative(self, xi, param):
        N = len(xi)
        m = len(xi[0])
        return [[0.] * m] * N
