"""Small numerical helpers for the asymmetric Clausius workflow."""

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


def bisection_root(
    func: Callable[[float], float | None],
    left: float,
    right: float,
    tol: float = 1.0e-10,
    max_iter: int = 300,
) -> float | None:
    """Find a scalar root inside `[left, right]`."""
    f_left = func(left)
    f_right = func(right)

    if f_left is None or f_right is None:
        return None
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        return None

    a = left
    b = right
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        f_c = func(c)
        if f_c is None:
            return None
        if abs(f_c) < tol or abs(b - a) < tol:
            return c
        if f_left * f_c <= 0.0:
            b = c
            f_right = f_c
        else:
            a = c
            f_left = f_c
    return 0.5 * (a + b)


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


def numerical_derivative(func: Callable[[float], float], x: float, h: float) -> float:
    """Return a centered finite-difference derivative."""
    return (func(x + h) - func(x - h)) / (2.0 * h)
