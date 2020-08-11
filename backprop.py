"""
    Example backpropagation implementation

    For each node function f_i
        - method to calculate value on input, f_i(x_i)
        - method to calculate derivative value on input, K_ij
        - method to calculate parameter derivative on input, xi_i

"""


def run_backprop_algorithm(params, graph, x, n_iter, eta):
    for i in range(n_iter):
        run_iteration(i, params, graph, x, eta)


def run_iteration(i, params, graph, x, eta):
    # run forward_prop iteration
    forward_store = run_forwardprop_iter(graph, x)
    # run backward prop iteration
    for param in params:
        del_p = run_backprop_iter(graph, param, forward_store)
        param.set_value(param.get_value() - eta * del_p)
        if i % 1000 == 0:
            print(f"updated params: '{param.name}': {param.value}")


def run_forwardprop_iter(graph, x):
    forward_store = {}
    graph.forward_prop(x, forward_store)
    return forward_store


def run_backprop_iter(graph, param, forward_store):
    # run backprop on the given parameter
    return graph.backward_prop(param, forward_store)

