import sys


class NodeFunction(object):

    # todo: accommodate real support for multiple child nodes

    def __init__(self, name, n_child_nodes):
        self.name = name
        self.n_child_nodes = n_child_nodes
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

    def derivative(self, xi, child_index):
        # jacobian value at xi wrt child node specified
        # by child_index
        pass

    def param_derivative(self, xi, param):
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
        # todo: need to handle case of multiple child nodes/inputs, possibly mixed
        if self.n_child_nodes == 1 and not self.children:
            forward_store[self.name] = leaf_x  # cache the input to this node
            return self.evaluate(leaf_x)
        elif self.n_child_nodes > 1:
            pass # to do, and other cases .....
        else:
            xi = []
            for child in self.children:
                xc = child.forward_prop(leaf_x, forward_store)
                xi.append(xc)
            if len(xi) == 1:
                forward_store[self.name] = xi[0]
                return self.evaluate(xi[0])
            forward_store[self.name] = xi
            return self.evaluate(xi)
            #self.evaluate(xc) for xc in xi]

    def _compute_beta(self, param, forward_store, ksi_store,
                      k_store):
        beta = [[0.]]
        xi = forward_store[self.name]
        # cache lookup
        if self.name in ksi_store:
            ksi = ksi_store[self.name]
        else:
            ksi = self.param_derivative(xi, param)
            ksi_store[self.name] = ksi
        for m in range(len(ksi)):
            for n in range(len(ksi[m])):
                beta[m][n] += ksi[m][n]
        #beta += ksi
        #print(f'ksi for node [{self.name}]: {ksi}')
        for child_index, child in enumerate(self.children):
            if param in child.all_params():
                # cache lookup
                if child.name in k_store:
                    k = k_store[child.name]
                else:
                    k = self.derivative(xi, child_index)
                    k_store[child.name] = k
                #print(f'k[{self.name}][{child.name}]: {k}')
                beta_c = child._compute_beta(param, forward_store,
                                             ksi_store, k_store)
                # matrix product between k and beta since each node multidimensional
                for m in range(len(beta)):
                    for n in range(len(beta[m])):
                        mm = 0.
                        for l in range(len(beta_c[m])):
                            mm += k[m][n][l] * beta_c[m][l]
                        beta[m][n] += mm

                #beta += k * beta_c
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
        bprop_out = self._compute_beta(param, forward_store,
                                  ksi_store,
                                  k_store)
        assert len(bprop_out) == 1, "backpropagation output not a scalar"
        assert len(bprop_out[0]) == 1, "backpropagation output not a scalar"
        return bprop_out[0][0]
