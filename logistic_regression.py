"""

    Logistic regression with node function constructs

    to do

"""
from param import Param
from sigmoid import Sigmoid
from projection import Projection


def build_graph():
    weights = []
    input_dim = 5
    for i in range(input_dim):
        weights.append(Param(f'weight_{i}', 1.))
    sigmoid = Sigmoid('sigmoid node')
    proj = Projection('projection node', weights)


def train():
    pass