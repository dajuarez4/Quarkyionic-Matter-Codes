"""Local numerical helpers for the quarkyonic sound-speed workflow."""

from __future__ import annotations

import math
from typing import Callable


def linspace(start: float, stop: float, count: int) -> list[float]:
    """Return `count` evenly spaced values from `start` to `stop`."""
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]

    step = (stop - start) / float(count - 1)
    return [start + step * float(i) for i in range(count)]


def simpson_integral(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Compute a 1D integral with Simpson's rule."""
    if n % 2 == 1:
        n += 1
    if b <= a:
        return 0.0

    x = linspace(a, b, n + 1)
    y = [float(func(xi)) for xi in x]
    h = (b - a) / float(n)

    total = y[0] + y[-1]
    total += 4.0 * sum(y[1:-1:2])
    total += 2.0 * sum(y[2:-1:2])
    return h * total / 3.0


def golden_section_min(
    func: Callable[[float], float],
    left: float,
    right: float,
    tol: float = 1.0e-8,
    max_iter: int = 300,
) -> tuple[float, float]:
    """Minimize a scalar function with the golden-section method."""
    gr = 0.5 * (math.sqrt(5.0) - 1.0)

    a = left
    b = right
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)

    x_min = 0.5 * (a + b)
    return x_min, func(x_min)


def first_derivative_from_grid(x: list[float], y: list[float]) -> list[float]:
    """Compute a first derivative on a 1D grid with finite differences."""
    if len(x) != len(y) or len(x) < 2:
        return []
    if len(x) == 2:
        dx = x[1] - x[0]
        slope = (y[1] - y[0]) / dx
        return [slope, slope]

    out = [0.0] * len(x)
    h0 = x[1] - x[0]
    h1 = x[-1] - x[-2]

    for i in range(len(x)):
        if i == 0:
            out[i] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h0)
        elif i == len(x) - 1:
            out[i] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * h1)
        else:
            dx = x[i + 1] - x[i - 1]
            out[i] = (y[i + 1] - y[i - 1]) / dx
    return out


def second_derivative_from_grid(x: list[float], y: list[float]) -> list[float]:
    """Compute a second derivative on a uniform 1D grid with finite differences."""
    if len(x) != len(y) or len(x) < 3:
        return []

    out = [0.0] * len(x)
    h0 = x[1] - x[0]
    h1 = x[-1] - x[-2]

    out[0] = (y[0] - 2.0 * y[1] + y[2]) / (h0 * h0)
    out[-1] = (y[-1] - 2.0 * y[-2] + y[-3]) / (h1 * h1)

    for i in range(1, len(x) - 1):
        h = 0.5 * (x[i + 1] - x[i - 1])
        out[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / (h * h)

    return out


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense linear system with Gaussian elimination."""
    n = len(rhs)
    aug = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-20:
            return [0.0] * n
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def smooth_local_polynomial(
    x: list[float],
    y: list[float],
    window_size: int,
    degree: int,
) -> list[float]:
    """Smooth a 1D curve with a local polynomial least-squares fit."""
    if len(x) != len(y) or len(x) == 0:
        return []

    if window_size < 3 or degree < 0 or len(x) < degree + 1:
        return y[:]

    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    out: list[float] = []
    for i in range(len(x)):
        left = max(0, i - half)
        right = min(len(x), i + half + 1)

        if right - left < degree + 1:
            if left == 0:
                right = min(len(x), degree + 1)
            else:
                left = max(0, len(x) - (degree + 1))

        x_local = [x[j] - x[i] for j in range(left, right)]
        y_local = [y[j] for j in range(left, right)]

        dim = degree + 1
        normal = [[0.0 for _ in range(dim)] for _ in range(dim)]
        rhs = [0.0 for _ in range(dim)]

        for x_value, y_value in zip(x_local, y_local):
            powers = [1.0]
            for _ in range(1, dim):
                powers.append(powers[-1] * x_value)

            for row in range(dim):
                rhs[row] += powers[row] * y_value
                for col in range(dim):
                    normal[row][col] += powers[row] * powers[col]

        coeffs = _solve_linear_system(normal, rhs)
        out.append(coeffs[0])

    return out
