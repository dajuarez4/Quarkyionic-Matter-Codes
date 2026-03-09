import numpy as np
import math

# ------------- help-functions 
def integrate_simpson(f, a, b, n=1000):
    h = (b - a) / n
    s = f(a) + f(b)
    x = a
    for i in range(1, n):
        x += h
        s += (4.0 if (i % 2) == 1 else 2.0) * f(x)
    return s * (h / 3.0)

def derivative(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("x and y must have same length and length >= 2")

    dydx = np.empty(n, dtype=float)

    # forward (first)
    dx = x[1] - x[0]
    if dx == 0:
        raise ValueError("Duplicate x values at indices 0 and 1.")
    dydx[0] = (y[1] - y[0]) / dx

    # central (interior)
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if dx == 0:
            raise ValueError(f"Duplicate x values around index {i}.")
        dydx[i] = (y[i + 1] - y[i - 1]) / dx

    # backward (last)
    dx = x[-1] - x[-2]
    if dx == 0:
        raise ValueError("Duplicate x values at the last two indices.")
    dydx[-1] = (y[-1] - y[-2]) / dx

    return dydx


phi = (math.sqrt(5) + 1) / 2

def gss(f, a, b, tol=1e-7, max_iter=10_000):
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = f(c)
    fd = f(d)

    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / phi
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / phi
            fd = f(d)
        it += 1

    return (a + b) / 2
