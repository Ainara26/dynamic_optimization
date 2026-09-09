import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).parent

def f(x):
    return 2*x-x**2

def c(x):
    x = np.concatenate([[0.1], x])
    x_next = x[1:]
    x_current = x[:-1]
    return f(x_current)-x_next

def J(x, args):
    phi = args
    capture = c(x)
    return -(phi*x[-1] + np.sum(np.sqrt(np.maximum(capture, 0.0))))

N = 100
EQUILIBRIUM = 0.5            # from Q2


def solve(phi, N=N):
    """Maximise phi*X_N + sum sqrt(C_t). Returns (X, C, J)."""
    guess = np.full(N, 0.15)
    cons = {'type': 'ineq', 'fun': c}
    bnds = [(0, 1)] * N
    res = minimize(fun=J, x0=guess, args=(phi,), bounds=bnds,
                    constraints=cons, method="SLSQP",
                    options={"maxiter": 500, "ftol": 1e-12})
    X = np.concatenate([[0.1], res.x])
    return X, c(res.x), -res.fun

def costate(c):
    return 1/(2*np.sqrt(c))

def df_dx(x):
    return 2 - 2*x

phi = 1.25

X, C, Jv = solve(phi)
lam = costate(C)
lhs = lam[0:N-1]
rhs = lam[1:N]*df_dx(X[1:N])
residual = np.max(np.abs(lhs-rhs))

print(residual)    
