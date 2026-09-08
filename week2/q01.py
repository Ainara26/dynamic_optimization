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

def J(x):
    capture = c(x)
    return -np.sum(np.sqrt(capture))

N = 100
x0 = np.full(N, 0.15)
cons = ({'type': 'ineq', 'fun': c})
bnds = [(0, 1)]*N
obj  = minimize(fun=J, x0=x0, bounds=bnds, constraints=cons, method="SLSQP")
print("J value:", -obj.fun)

# plots
X = np.concatenate([[0.1], obj.x])      # X_0..X_N       length N+1
C = c(obj.x)                            # C_0..C_{N-1}   length N
t_state = np.arange(N + 1)              # 0..N
t_harv = np.arange(N)                   # 0..N-1

# the flat middle of the trajectory, read off the solution itself
mid = slice(N // 4, 3 * N // 4)
plateau_x = np.median(X[mid])
plateau_c = np.median(C[mid])

fig, axd = plt.subplot_mosaic([["ts_x", "phase"],
                               ["ts_c", "phase"]], figsize=(12, 7))

# time series: the state
ax = axd["ts_x"]
ax.plot(t_state, X, "o-", ms=3.5, lw=1.2, color="tab:blue")
ax.axhline(plateau_x, color="0.5", ls=":", lw=1.2,
           label=f"plateau $x \\approx {plateau_x:.3f}$")
ax.set_ylabel("stock $X_t$")
ax.set_title(f"Q1: optimal fishery, $N={N}$, $X_0=0.1$   ($J={-obj.fun:.4f}$)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# time series: the harvest
ax = axd["ts_c"]
ax.plot(t_harv, C, "o-", ms=3.5, lw=1.2, color="tab:red")
ax.axhline(plateau_c, color="0.5", ls=":", lw=1.2,
           label=f"plateau $c \\approx {plateau_c:.3f}$")
ax.set_xlabel("$t$")
ax.set_ylabel("harvest $C_t$")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# the (x, c)-plane
ax = axd["phase"]
xs = np.linspace(0, 1, 300)
ax.plot(xs, f(xs) - xs, "k--", lw=1.4,
        label=r"$c=f(x)-x$  (stock unchanged)")
ax.plot(X[:-1], C, "o-", ms=4, lw=1.0, color="tab:purple", alpha=0.85,
        label="trajectory")
ax.plot(X[0], C[0], "s", ms=11, color="tab:green", mec="black",
        label=f"start $(X_0,C_0)$", zorder=5)
ax.plot(X[-2], C[-1], "*", ms=17, color="tab:orange", mec="black",
        label="last period", zorder=5)
ax.plot(plateau_x, plateau_c, "D", ms=9, color="crimson", mec="black",
        label=f"plateau $({plateau_x:.3f},\\,{plateau_c:.3f})$", zorder=6)
ax.set_xlabel("stock $x$")
ax.set_ylabel("harvest $c$")
ax.set_title("the $(x,c)$-plane")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()
out = HERE / f"q01_N{N}.png"
fig.savefig(out, dpi=150)
print("saved", out.name)
