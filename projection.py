from node_function import NodeFunction


class Projection(NodeFunction):

    """
    x |-> <x, w>
    """

    def __init__(self, name, weights):
        super().__init__(name, n_child_nodes=1)
        self.weights = weights
        self.weights_index = {}
        for i, weight in enumerate(weights):
            self.direct_params.add(weight)
            self.weights_index[id(weight)] = i

    def evaluate(self, xi):
        output = []
        for row in xi:
            proj_sum = 0.
            for j, e in enumerate(row):
                proj_sum += e * self.weights[j].value
            out_row = [proj_sum]
            output.append(out_row)
        return output

    def derivative(self, xi, child_index):
        N = len(xi)
        m = len(xi[0])
        out_mat = [[[0] * m] * 1] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][0][j] = self.weights_index[j]
        return out_mat

    def param_derivative(self, xi, param):
        k = self.weights_index[id(param)]
        p_deriv = []
        for row in xi:
            p_deriv.append([row[k]])
        return p_deriv
