def compute_nth_derivative(X, n, dt):
    for i in range(n):
        X = (X[:, 1:] - X[:, :-1]) / dt
    return X

