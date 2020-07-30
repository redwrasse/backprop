from node_function import NodeFunction


class Mult(NodeFunction):

    def __init__(self, name, param):
        super().__init__(name, n_inputs=1)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        return xi * self.param.value

    def derivative(self, xi, input_index):
        return self.param.value

    def param_derivative(self, xi, param):
        return xi