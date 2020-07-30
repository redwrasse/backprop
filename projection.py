from node_function import NodeFunction


class Projection(NodeFunction):

    """
    x |-> <x, w>
    """

    def __init__(self, name, n_inputs, weights):
        super().__init__(name, n_inputs=n_inputs)
        assert len(n_inputs) == len(weights)
        self.weights = weights
        self.weights_index = {}
        for i, weight in enumerate(weights):
            self.direct_params.add(weight)
            self.weights_index[id(weight)] = i

    def evaluate(self, xi):
        return sum(xi[k] * self.weights[k].value for k in range(self.n_inputs))

    def derivative(self, xi, input_index):
        return self.weights[input_index].value

    def param_derivative(self, xi, param):
        k = self.weights_index[id(param)]
        return xi[k]
