import pathlib

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

A = -1
B = 1
C = 1

def func_vec(x, y):
    return 0.5 * x @ A @ x + x @ B @ y + 0.5 * y @ C @ y

def func(x, y):
    return 0.5*A*x**2 + B*x*y + 0.5*C*y**2

def g_func(x):
    return 0.5 * x**2 *(A- B**2/C) 

def y_star(x):
    return -1/C*B*x

x_plot = np.linspace(-3, 3, 400)
x_vals = np.linspace(-3, 3, 20)
y_vals = y_star(x_vals)


def solve_J(x):
    x0 = np.array([0.0])
    a = -1
    b = 1
    c = 1
    args = a, b, c
    obj = lambda y: func(x, y)
    res = minimize(obj, x0)
    return res.fun, res.x[0]


HERE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------- Q6 plot 1
fig, ax = plt.subplots(figsize=(8, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(y_vals)))
for y, col in zip(y_vals, colors):
    ax.plot(x_plot, func(x_plot, y), color=col, lw=1.0)

ax.plot(x_plot, g_func(x_plot), "k-", lw=3, label=r"$g(x)=\min_y f(x,y)$")
# each parabola touches g where that y is the optimal choice, i.e. at x_vals
ax.plot(x_vals, g_func(x_vals), "o", ms=5, color="crimson", zorder=5,
        label="tangency points")

ax.set_xlabel("$x$")
ax.set_ylabel("$f(x,y)$")
ax.set_title(rf"Q6: $f(x,y)$ for {len(y_vals)} values of $y$, "
             rf"and $g(x)$   ($A={A}$, $B={B}$, $C={C}$)")
ax.legend(loc="lower center")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "q06_envelope.png", dpi=150)
print(f"saved q06_envelope.png")

# ---------------------------------------------------------------- Q6 plot 2
xg = np.linspace(-3, 3, 400)
yg = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(xg, yg)
Z = func(X, Y)

fig, ax = plt.subplots(figsize=(7, 6.5))
lim = np.abs(Z).max()
cs = ax.contour(X, Y, Z, levels=np.linspace(-lim, lim, 25),
                cmap="RdBu_r", linewidths=0.9)
fig.colorbar(cs, ax=ax, label="$f(x,y)$")

ax.plot(xg, y_star(xg), "k-", lw=2.5, label=r"optimal $y=y^*(x)$")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")
ax.set_title("Q6: contour of $f(x,y)$ with the optimal $y$ for each $x$")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "q06_contour.png", dpi=150)
print(f"saved q06_contour.png")

