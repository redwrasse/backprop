from node_function import NodeFunction


class Mult(NodeFunction):
    """ vectorized multiplication by a scalar parameter on a batch
    of inputs

    x_ij |-> c * x_ij
    """
    def __init__(self, name,  param, n_inputs):
        super().__init__(name, n_inputs=n_inputs)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        output = []
        for row in xi:
            out_row = []
            for e in row:
                out_row.append(e * self.param.value)
            output.append(out_row)
        return output

    def derivative(self, xi, input_index):
        N = len(xi)
        m = len(xi[0])
        out_vec = [[0] * m] * N
        for i in range(N):
            out_vec[i][input_index] = self.param.value
        return out_vec

    def param_derivative(self, xi, param):
        return xi
