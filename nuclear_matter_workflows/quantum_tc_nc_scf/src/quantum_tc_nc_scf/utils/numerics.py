"""Basic numerical helper functions."""

from __future__ import annotations

from typing import Callable, List, Tuple


def linspace(start: float, stop: float, count: int) -> List[float]:
    """Return `count` evenly spaced values in `[start, stop]`."""
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]

    step = (stop - start) / float(count - 1)
    return [start + step * float(i) for i in range(count)]


def simpson_weights(a: float, b: float, n: int) -> Tuple[List[float], List[float]]:
    """Return the integration grid and weights for Simpson's rule."""
    if n % 2 == 1:
        n += 1

    x = linspace(a, b, n + 1)
    h = (b - a) / float(n)

    w = [1.0] * (n + 1)
    for i in range(1, n, 2):
        w[i] = 4.0
    for i in range(2, n, 2):
        w[i] = 2.0
    w = [weight * h / 3.0 for weight in w]

    return x, w


def numerical_derivative_scalar(func: Callable[[float], float], x: float, h: float) -> float:
    """Centered finite-difference derivative for scalar functions."""
    return (func(x + h) - func(x - h)) / (2.0 * h)

