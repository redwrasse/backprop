from node_function import NodeFunction
import math


class Sigmoid(NodeFunction):

    """ Treating sigmoid as a node function with no parameters """

    def __init__(self, name):
        super().__init__(name, n_child_nodes=1)

    def evaluate(self, xi):
        output = []
        for row in xi:
            out_row = []
            for e in row:
                out_row.append(math.exp(e) / (1. + math.exp(e)))
            output.append(out_row)
        return output

    def derivative(self, xi, child_index):
        # phi' = phi * (1 - phi)
        phi_v = self.evaluate(xi)
        output = []
        for row in phi_v:
            out_row = []
            for e in row:
                out_row.append(e * (1. - e))
            output.append(out_row)
        return output

    def param_derivative(self, xi, param):
        N = len(xi)
        m = len(xi[0])
        return [[0.] * m] * N
