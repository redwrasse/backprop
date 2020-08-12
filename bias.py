from node_function import NodeFunction


class Bias(NodeFunction):
    """
    x_i |-> b_i + x_i, b_i a trainable vector
    """

    def __init__(self, name, biases):
        super().__init__(name, n_child_nodes=1)
        self.biases = biases
        self.biases_index = {}
        for i, bias in enumerate(self.biases):
            self.direct_params.add(bias)
            self.biases_index[id(bias)] = i

    def evaluate(self, xi):
        output = []
        for row in xi:
            out_row = []
            for j, e in enumerate(row):
                out_row.append(e + self.biases[j].value)
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
        p_deriv = [[0.] * m] * N
        k = self.biases_index[id(param)]
        for i in range(N):
            p_deriv[i][k] = 1.
        return p_deriv
