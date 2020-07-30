"""
    Suppose a single input x in R. Suppose the functional to be
    optimized is F[a] = xa^2 + a

    F'[a] = 2xa + 1

    2xa + 1 = 0
        => a = -1 / (2x) at minimum

    Hence for x = 4, algorithm should converge to a = - 1 / 8. = -0.125


Function composition graph. Assume elementary operations:
Add, Mult.

    F[a] = add_a ( mult_a ( mult_a( (x))) ))


So graph is:


  add_a
    |
 mult_a
    |
 mult_a
    |
 (input x)


The terms are

add_a
-----
ksi = 1
K_[this][child] = 1

mult_a
------
ksi = x
K_[this][child] = a


Output
~~~~~~

registering nodes ...
registered: child [multiplication node 2] of [multiplication node 1] at index 0.
registered: child [multiplication node 1] of [addition node] at index 0.
finished registering nodes.
registering param dependencies with nodes ...
built indirect param set [] for node [multiplication node 2]
built indirect param set ['a'] for node [multiplication node 1]
built indirect param set ['a'] for node [addition node]
finished registering params with nodes.
updated params: 'a': 0.991
updated params: 'a': -0.1246374763459712
updated params: 'a': -0.124999882237097
updated params: 'a': -0.12499999996174568
updated params: 'a': -0.12499999999998755
updated params: 'a': -0.12499999999999914
updated params: 'a': -0.12499999999999914
updated params: 'a': -0.12499999999999914
updated params: 'a': -0.12499999999999914
updated params: 'a': -0.12499999999999914
"""


from add import Add
from backprop import run_backprop_algorithm
from mult import Mult
from param import Param


def sample_graph():
    # build computation graph
    params = []
    a = Param('a', 1.)
    params.append(a)
    add = Add('addition node', a)
    mult = Mult('multiplication node 1', a)
    mult2 = Mult('multiplication node 2', a)
    mult.add_child(mult2)
    add.add_child(mult)
    graph = add
    graph.set_complete()
    return params, graph


def run_example1():
    params, graph = sample_graph()
    run_backprop_algorithm(params, graph,
                           n_iter=10**4,
                           eta=1e-3)
    a = params[0].value
    assert abs(a - (-1 / 8.)) < 10e-4, "a did not converge to -1 / 8."


if __name__ == "__main__":
    run_example1()
