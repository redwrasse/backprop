"""
    Example backpropagation implementation

    For each node function f_i
        - method to calculate value on input, f_i(x_i)
        - method to calculate derivative value on input, K_ij
        - method to calculate parameter derivative on input, xi_i

    Suppose a single input x in R. Suppose the functional to be
    optimized is F[a] = xa^2 + a

    2xa + 1 = 0
        => a = -1 / (2x) at minimum


Function composition graph. Assume elementary operations:
Add, Mult.

    F[a] = Add_a ( Mult_a^2 (x) ))


So graph is:

 x ->  Mult_a^2 - Add_a

"""
import sys


def functional(x):
    def func(a):
        return x * a ** 2 + a
    return func


def direct_gradient_descent():
    """
    for comparison:
    optimization by direct gradient descent on F,
    computing parameter derivatives numerically
    """
    x = 4
    F = functional(x)

    # initial parameter
    a = 1.
    # learning rate
    eta = 1e-1
    # number of gradient descent iterations
    n_iters = 10 ** 4
    eps = 0.001
    # should converge to minimum at -1 / (2x)
    # = -1 / 8 = -0.125
    for i in range(n_iters):
        dF_a = (F(a + eps) - F(a)) / eps
        a += - eta * dF_a
        if i % 10 ** 3 == 0:
            print(f'i: {i} a: {a}')
    assert abs(a - (-1 / 8.)) < 10e-4, "a did not converge to -1 / 8."


class NodeFunction(object):

    def __init__(self, name):
        self.name = name
        self.children = []
        self.is_complete = False
        self.direct_params = set()
        self.indirect_params = set()

    def set_complete(self):
        print('registering nodes ...')
        self._register()
        print('finished registering nodes.')
        print('registering param dependencies with nodes ...')
        self.build_indirect_params()
        print('finished registering params with nodes.')

    def all_params(self):
        return self.direct_params.union(self.indirect_params)

    def _add_indirect_param(self, param):
        self.indirect_params.add(param)

    def build_indirect_params(self):
        for child in self.children:
            child_indirect_params = child.build_indirect_params()
            self.indirect_params = self.indirect_params.union(child_indirect_params)
        print(f'built indirect param set {[_.name for _ in self.indirect_params]} for node [{self.name}]')
        return self.direct_params.union(self.indirect_params)

    def _register(self):
        for i, child in enumerate(self.children):
            child._register()
            print(f'registered: child [{child.name}] of [{self.name}] at index {i}.')
        self.is_complete = True

    def evaluate(self, xi):
        # value
        pass

    def derivative(self, xi):
        # jacobian value
        pass

    def param_derivative(self, xi):
        # direct param derivative
        pass

    def add_child(self, child):
        self.children.append(child)

    # get evaluation points by forward prop
    # and store in dict forward_store
    def forward_prop(self, leaf_x, forward_store):
        if not self.is_complete:
            print('error: graph not complete. exiting')
            sys.exit(0)
        if not self.children:
            forward_store[self.name] = leaf_x
            return self.evaluate(leaf_x)
        else:
            xi = []
            for child in self.children:
                xc = child.forward_prop(leaf_x, forward_store)
                xi.append(xc)
            if len(xi) == 1:
                forward_store[self.name] = xi[0]
                return self.evaluate(xi[0])
            forward_store[self.name] = xi
            return [self.evaluate(xc) for xc in xi]

    def _compute_beta(self, param, forward_store):
        beta = 0.
        xi = forward_store[self.name]
        ksi = self.param_derivative(xi)
        beta += ksi
        for child in self.children:
            if param in child.all_params():
                k = child.derivative(xi)
                beta_c = child._compute_beta(param, forward_store)
                beta += k * beta_c
        return beta

    def backward_prop(self, param, forward_store):
        print(f"running backprop from root node [{self.name}] on param '{param.name}' ... ")
        if not self.is_complete:
            print('error: graph not complete. exiting')
            sys.exit(0)
        print('finished backprop.')
        return self._compute_beta(param, forward_store)


class Add(NodeFunction):

    def __init__(self, name, param):
        super().__init__(name)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        return xi + self.param.value

    def derivative(self, xi):
        return 1.

    def param_derivative(self, xi):
        return 1.


class Mult(NodeFunction):

    def __init__(self, name, param):
        super().__init__(name)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        return xi * self.param.value

    def derivative(self, xi):
        return self.param.value

    def param_derivative(self, xi):
        return xi


class Param(object):

    def __init__(self, name, init_value):
        self.name = name
        self.value = init_value

    def set_value(self, value):
        self.value = value


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


def run_forward_prop(graph, x):
    forward_store = {}
    graph.forward_prop(x, forward_store)
    return forward_store


def run_back_prop(graph, param, forward_store):
    # run backprop on the given parameter
    graph.backward_prop(param, forward_store)


def run_algorithm():

    params, graph = sample_graph()
    # run forward_prop
    print('running forward prop ...')
    x = 4.
    forward_store = run_forward_prop(graph, x)
    print('finished forward prop.')
    print(f'cached evaluation points from forward prop: {forward_store}')

    # run backward prop
    # to do
    for param in params:
        run_back_prop(graph, param, forward_store)


if __name__ == "__main__":
    # direct_gradient_descent()
    run_algorithm()
