import pathlib

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent   # so paths work from any directory

def func(x, u):
    return (x-u)**2

def solve_J(x):
    x0 = np.array([0.5])
    obj = lambda u: func(x, u) # x is fixed and just u varies
    res = minimize(obj, x0, bounds=[(0, 1)], method="L-BFGS-B")
    # minimize the distance between x and u, being u bounded between 0 and 1
    return res.fun, res.x[0]

xs = np.linspace(-1, 2, 300)
J = np.zeros_like(xs) # J will define how badly I missed from the chosen x
mu = np.zeros_like(xs) # this will describe the 'u' we choose

for i, x in enumerate(xs):
    J[i], mu[i] = solve_J(x)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,7))

ax1.plot(xs, mu, lw=2, color="tab:blue")
ax1.set_ylabel(r"$\mu(x)$")
ax1.set_title(r"Q4:  $J(x)=\min_{u\in[0,1]}(x-u)^2$,   $\mu(x)=\arg\min$")
ax1.grid(alpha=0.3)

ax2.plot(xs, J, lw=2, color="tab:red")
ax2.set_ylabel(r"$J(x)$")
ax2.set_xlabel(r"$x$")
ax2.grid(alpha=0.3)

# u is trapped in [0,1]; those boundaries are where the behaviour changes
for ax in (ax1, ax2):
    ax.axvline(0.0, color="0.6", ls=":", lw=1)
    ax.axvline(1.0, color="0.6", ls=":", lw=1)

fig.tight_layout()
fig.savefig(HERE / "q04.png", dpi=150)

lhs = np.gradient(J, xs) #numerical derivative of J
rhs = 2*(xs-mu)
print("max difference:", np.max(np.abs(lhs-rhs)))
