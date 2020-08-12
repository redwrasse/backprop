"""

    Linear regression. The elementary functions considered are addition, matrix multiplication, square. y’ = Ax + b = sum_b(mm_A(x)). The loss is l = 1/N sum_i (y - y’)^2. The parameter set is (A, b). The functional to be minimized F is over those functions F = F[mm_A, sum_b]. Considered over parameters it is F(A, b). Clearly F is the loss value over training inputs.

A possible composition graph



 mult_1/N
    |
  square
    |
 subtract - (input y)
    |
  sum_b
    |
   mm_A
    |
 (input x)

"""
from param import Param
from subtract import Subtract
from normsquare import NormSquare
from bias import Bias
from projection import Projection
from constantmult import ConstMult
from average import Average
from backprop import run_backprop_algorithm


def build_graph():

    input_dim = 3
    N = 3


    weights = []
    biases = []
    for i in range(input_dim):
        weights.append(Param(f'weight_{i}', 1.))
        biases.append(Param(f'bias_{i}', 1.))

    params = weights + biases

    proj = Projection('projection node', weights)
    sub = Subtract("subtract node")
    normsq = NormSquare("norm square node")
    bia = Bias("bias node", biases)
    cmult = ConstMult("1/N const mult", 1. / N)
    cmult.add_child(normsq)
    normsq.add_child(sub)
    sub.add_child(bia) # how to add input y as child / second input?
    bia.add_child(proj)

    graph = cmult
    graph.set_complete()
    return params, graph


def train():
    params, graph = build_graph()
    input_dim = 3
    N = 3

    x = [
        [1., 2., 3.],
        [2., 3., 4.],
        [5., 6., 7.],
    ]
    A = [3., 3, 3.]
    b = [1., 1., 1.]
    y = []

    for i in range(N):
        output_row = []
        for j in range(input_dim):
            output_row.append(A[j] * x[i][j] + b[j])
        y.append(output_row)

    run_backprop_algorithm(params, graph, [x, y],
                           n_iter=10**4,
                           eta=1e-3)




if __name__ == "__main__":
    train()
