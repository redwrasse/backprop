"""
    Example backpropagation implementation

    For each node function f_i
        - method to calculate value on input, f_i(x_i)
        - method to calculate derivative value on input, K_ij
        - method to calculate parameter derivative on input, xi_i

    Suppose a single input x in R. Suppose the functional to be
    optimized is F[a] = xa^2 + a

    F'[a] = 2xa + 1

    2xa + 1 = 0
        => a = -1 / (2x) at minimum

    Hence for x = 4, algorithm should converge to a = - 1 / 8. = -0.125


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

    def __init__(self, name, n_inputs):
        self.name = name
        self.n_inputs = n_inputs
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
        all_names = set()
        for i, child in enumerate(self.children):
            c_all_names = child._register()
            if not all_names.isdisjoint(c_all_names):
                print('error: duplicate node names. exiting')
                sys.exit(0)
            all_names = all_names.union(c_all_names)
            print(f'registered: child [{child.name}] of [{self.name}] at index {i}.')
        self.is_complete = True
        return all_names

    def evaluate(self, xi):
        # value
        pass

    def derivative(self, xi, input_index):
        # jacobian value at xi wrt input specified
        # by input index
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

    def _compute_beta(self, param, forward_store, ksi_store,
                      k_store):
        beta = 0.
        xi = forward_store[self.name]
        # cache lookup
        if self.name in ksi_store:
            ksi = ksi_store[self.name]
        else:
            ksi = self.param_derivative(xi)
            ksi_store[self.name] = ksi
        beta += ksi
        #print(f'ksi for node [{self.name}]: {ksi}')
        for input_index, child in enumerate(self.children):
            if param in child.all_params():
                # cache lookup
                if child.name in k_store:
                    k = k_store[child.name]
                else:
                    k = self.derivative(xi, input_index)
                    k_store[child.name] = k
                #print(f'k[{self.name}][{child.name}]: {k}')
                beta_c = child._compute_beta(param, forward_store,
                                             ksi_store, k_store)
                beta += k * beta_c
        #print(f'beta for node [{self.name}]: {beta}')
        return beta

    def backward_prop(self, param, forward_store):
        #print(f"running backprop from root node [{self.name}] on param '{param.name}' ... ")
        if not self.is_complete:
            print('error: graph not complete. exiting')
            sys.exit(0)
        #print('finished backprop.')
        ksi_store = {}
        k_store = {}
        return self._compute_beta(param, forward_store,
                                  ksi_store,
                                  k_store)


class Add(NodeFunction):

    def __init__(self, name, param):
        super().__init__(name, n_inputs=1)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        return xi + self.param.value

    def derivative(self, xi, input_index):
        return 1.

    def param_derivative(self, xi):
        return 1.


class Mult(NodeFunction):

    def __init__(self, name, param):
        super().__init__(name, n_inputs=1)
        self.param = param
        self.direct_params.add(param)

    def evaluate(self, xi):
        return xi * self.param.value

    def derivative(self, xi, input_index):
        return self.param.value

    def param_derivative(self, xi):
        return xi


class Param(object):

    def __init__(self, name, init_value):
        self.name = name
        self.value = init_value

    def get_value(self):
        return self.value

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


def run_forwardprop_iter(graph, x):
    forward_store = {}
    graph.forward_prop(x, forward_store)
    return forward_store


def run_backprop_iter(graph, param, forward_store):
    # run backprop on the given parameter
    return graph.backward_prop(param, forward_store)


def run_backprop_algorithm():

    # build graph and initial parameter values
    params, graph = sample_graph()
    # learning rate
    eta = 1e-3
    # number of iterations
    n_iter = 10**4
    for i in range(n_iter):
        run_iteration(i, params, graph)

    a = params[0].value
    assert abs(a - (-1 / 8.)) < 10e-4, "a did not converge to -1 / 8."


def run_iteration(i, params, graph):
    # run forward_prop iteration
    x = 4.
    forward_store = run_forwardprop_iter(graph, x)
    # run backward prop iteration
    eta = 1e-3
    for param in params:
        del_p = run_backprop_iter(graph, param, forward_store)
        param.set_value(param.get_value() - eta * del_p)
        if i % 1000 == 0:
            print(f"updated params: '{param.name}': {param.value}")


if __name__ == "__main__":
    # direct_gradient_descent()
    run_backprop_algorithm()
