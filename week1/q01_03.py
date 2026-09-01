"""Session 1: constrained and unconstrained optimization in the plane.

    h(x, y) = (x - 1)^2 + b (y - e^x)^2,    b = 10

Q1  analytic minimum (done by hand: unique global min at (1, e), h = 0)
    plus a contour plot marking it
Q2  the same minimum, found numerically
Q3  J(s) = min{ h(x,y) | x^2 + y^2 = s }: J(1), the multiplier mu, and a
    numerical check of J(1+eps) ~= J(1) - mu*eps

Run:  uv run python week1/q01_03.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

B = 10.0

HERE = pathlib.Path(__file__).parent


def h(x, y):
    """Evaluate h at (x, y). Works elementwise, so it accepts a meshgrid."""
    return (x - 1) ** 2 + B * (y - np.exp(x)) ** 2


def h_vec(v):
    """Same function, but taking the two variables as one array (for scipy)."""
    x, y = v
    return h(x, y)


def plot_contours(ax, xlim, ylim, n=400):
    """Draw log-spaced contours of h on ax, with a colorbar. Returns Z.

    h spans several orders of magnitude, so evenly spaced levels would bunch
    almost every line at the edges; geomspace gives each decade equal weight.
    It must start above 0 because min(h) = 0 and log(0) is undefined.
    """
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = h(X, Y)

    levels = np.geomspace(1e-2, Z.max(), 30)
    cs = ax.contour(X, Y, Z, levels=levels, norm="log",
                    cmap="viridis", linewidths=0.8)
    # default log ticks come out as 1.028e+02 etc, so pin them to decades
    ticks = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    cb = ax.figure.colorbar(cs, ax=ax, label="$h(x,y)$", ticks=ticks)
    cb.ax.set_yticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in ticks])

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return Z


# ------------------------------------------------------------------ Q1
def question1():
    """Contour plot of h, with the analytic minimum (1, e) marked."""
    # Narrow x-range, because e^x grows fast: at x = 2 the valley is already
    # at y = 7.4.
    fig, ax = plt.subplots(figsize=(7, 7))
    Z = plot_contours(ax, (-0.5, 2.0), (-1.0, 5.0))
    print(f"Q1  Z range over the window: {Z.min():.3e} to {Z.max():.3e}")

    ax.plot(1.0, np.e, "r*", ms=18, mec="black", mew=0.8,
            label=r"minimum $(1,\,e)$,  $h=0$")

    ax.set_title(rf"$h(x,y)=(x-1)^2+{B:g}\,(y-e^x)^2$")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = HERE / "q01_contour.png"
    fig.savefig(out, dpi=150)
    print(f"Q1  saved {out.name}")


# ------------------------------------------------------------------ Q2
def question2():
    """Solve min h numerically."""
    x0 = np.array([0.0, 0.0])
    res = minimize(h_vec, x0, method="BFGS")

    print(f"\nQ2  x* = {res.x}")
    print(f"Q2  h(x*) = {res.fun:.3e}")


# ------------------------------------------------------------------ Q3
def solve_J(s):
    """Minimise h on the circle x^2 + y^2 = s. Returns the scipy result."""
    con = NonlinearConstraint(lambda v: v[0]**2 + v[1]**2, s, s)
    x0 = np.array([0.0, np.sqrt(s)])          # a point already ON the circle
    return minimize(h_vec, x0, method="trust-constr", constraints=[con],
                    options={"gtol": 1e-12, "xtol": 1e-14})


def plot_q3(s, minimiser):
    """Contours of h, the constraint circle x^2+y^2 = s, and the optimum."""
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    plot_contours(ax, (-1.6, 2.0), (-1.6, 3.4))

    th = np.linspace(0, 2 * np.pi, 400)
    r = np.sqrt(s)
    ax.plot(r * np.cos(th), r * np.sin(th), "k-", lw=1.8,
            label=rf"constraint  $x^2+y^2={s:g}$")
    ax.plot(*minimiser, "ro", ms=11, mec="black", mew=0.9,
            label=rf"optimum  $({minimiser[0]:.4f},\,{minimiser[1]:.4f})$")

    ax.set_aspect("equal")                    # so the circle looks circular
    ax.set_title(rf"Q3: minimum of $h$ on the circle $x^2+y^2={s:g}$")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = HERE / "q03_constrained.png"
    fig.savefig(out, dpi=150)
    print(f"Q3  saved {out.name}")


def question3():
    """J(s) = min{ h(x,y) | x^2 + y^2 = s }; find J(1) and its multiplier."""
    res = solve_J(1)
    J_1 = res.fun
    minimiser = res.x
    multiplier = res.v[0][0]
    print("\nJ(1):", J_1)
    print("minimiser:", minimiser)
    print("multiplier:", multiplier)

    eps = 0.01
    res_eps = solve_J(1 + eps)
    J_1_eps = res_eps.fun
    print("J(1+eps):", J_1_eps)
    rhs = J_1 - multiplier * eps
    print("rhs:", rhs)
    print("difference:", J_1_eps - rhs)

    plot_q3(1.0, minimiser)


def main():
    question1()
    question2()
    question3()


if __name__ == "__main__":
    main()
