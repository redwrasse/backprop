from node_function import NodeFunction


class Average(NodeFunction):
    """ average over all input vector elements
    no parameters to be trained
    """
    def __init__(self, name, n_inputs):
        super().__init__(name, n_inputs=n_inputs)

    def evaluate(self, xi):
        N = len(xi)
        m = len(xi[0])
        return 1. / (N * m) * sum(a for e in xi for a in e)

    def derivative(self, xi, input_index):
        N = len(xi)
        m = len(xi[0])
        return [[1. / (N * m)] * m] * N

    def param_derivative(self, xi, param):
        N = len(xi)
        m = len(xi[0])
        return [[0.]]
