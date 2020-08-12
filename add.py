from node_function import NodeFunction


class Add(NodeFunction):
    """ vectorized addition with a scalar parameter on a batch
      of inputs

      x_ij |-> a + x_ij
      """
    def __init__(self, name, param):
        super().__init__(name, n_child_nodes=1)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        output = []
        for row in xi:
            out_row = []
            for e in row:
                out_row.append(e + self.param.value)
            output.append(out_row)
        return output

    def derivative(self, xi, child_index):
        # identity matrix for each row
        N = len(xi)
        m = len(xi[0])
        out_mat = [[[0] * m] * m] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][j][j] = 1.
        return out_mat

    def param_derivative(self, xi, param):
        N = len(xi)
        m = len(xi[0])
        return [[1.] * m] * N

