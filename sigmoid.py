from node_function import NodeFunction
import math


class Sigmoid(NodeFunction):

    """ Treating sigmoid as a node function with no parameters """

    def __init__(self, name):
        super().__init__(name, n_inputs=1)

    def evaluate(self, xi):
        return math.exp(xi) / (1. + math.exp(xi))

    def derivative(self, xi, input_index):
        # phi' = phi * (1 - phi)
        phi_v = self.evaluate(xi)
        return phi_v * (1. - phi_v)

    def param_derivative(self, xi, param):
        return 0.
