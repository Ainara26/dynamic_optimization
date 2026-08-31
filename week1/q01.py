"""Session 1, Q1: contour plot of h, with the minimum marked.

    h(x, y) = (x - 1)^2 + b (y - e^x)^2,    b = 10

The analytical half is already done by hand: unique global minimum at (1, e)
where h = 0. This script is the plotting half.

Run:  uv run python week1/q01.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

B = 10.0


def h(x, y):
    """Evaluate h at (x, y). Works elementwise, so it accepts a meshgrid."""
    return (x - 1) ** 2 + B * (y - np.exp(x)) ** 2


def main():
    # --- 1. build the grid ------------------------------------------------
    # Narrow x-range, because e^x grows fast: at x = 2 the valley is already
    # at y = 7.4. 400 points per axis keeps the contour lines smooth.
    x = np.linspace(-0.5, 2.0, 400)
    y = np.linspace(-1.0, 5.0, 400)
    X, Y = np.meshgrid(x, y)

    # --- 2. evaluate ------------------------------------------------------
    Z = h(X, Y)
    print(f"Z range: {Z.min():.3e} to {Z.max():.3e}")

    # --- 3. contour plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    levels = np.geomspace(1e-2, Z.max(), 30)
    cs = ax.contour(X, Y, Z, levels=levels, norm="log",
                    cmap="viridis", linewidths=0.8)
    # default log ticks come out as 1.028e+02 etc, so pin them to decades
    ticks = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    cb = fig.colorbar(cs, ax=ax, label="$h(x,y)$", ticks=ticks)
    cb.ax.set_yticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in ticks])

    # --- 4. mark the minimum ---------------------------------------------
    ax.plot(1.0, np.e, "r*", ms=18, mec="black", mew=0.8,
            label=r"minimum $(1,\,e)$,  $h=0$")

    # --- 5. finish --------------------------------------------------------
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(rf"$h(x,y)=(x-1)^2+{B:g}\,(y-e^x)^2$")
    ax.legend(loc="upper left")
    ax.set_ylim(y[0], y[-1])
    fig.tight_layout()

    out = pathlib.Path(__file__).parent / "q01_contour.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
