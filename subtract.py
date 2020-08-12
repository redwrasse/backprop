from node_function import NodeFunction


class Subtract(NodeFunction):

    """
     Difference of two vectors, given as distinct child nodes
     Subtract second child from first child
    """
    def __init__(self, name):
        super().__init__(name, n_child_nodes=2)

    def evaluate(self, xi):
        # in this case because there are two child nodes,
        # xi is expected to be an array of two inputs
        # those inputs each having the expected form
        # of a list of N vectors.
        assert len(xi) == self.n_child_nodes
        xa, xb = xi[0], xi[1]
        assert len(xa) == len(xb)
        N = len(xa)
        assert len(xa[0]) == len(xb[0])
        m = len(xa[0])
        output = []
        for i in range(N):
            output_row = []
            for j in range(m):
                output_row.append(xa[i][j] - xb[i][j])
            output.append(output_row)
        return output

    def derivative(self, xi, child_index):
        # derivative is pos. identity matrix wrt first child,
        # neg. identity matrix wrt second child
        # identity matrix for each row
        parity = 1. if child_index == 0 else -1.
        assert len(xi) == self.n_child_nodes
        xa, xb = xi[0], xi[1]
        assert len(xa) == len(xb)
        N = len(xa)
        assert len(xa[0]) == len(xb[0])
        m = len(xa[0])
        out_mat = [[[0] * m] * m] * N
        for i in range(N):
            for j in range(m):
                out_mat[i][j][j] = parity
        return out_mat

    def param_derivative(self, xi, param):
        assert len(xi) == self.n_child_nodes
        xa, xb = xi[0], xi[1]
        assert len(xa) == len(xb)
        N = len(xa)
        assert len(xa[0]) == len(xb[0])
        m = len(xa[0])
        return [[0.] * m] * N




