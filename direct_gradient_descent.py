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


def functional(x):
    def func(a):
        return x * a ** 2 + a
    return func