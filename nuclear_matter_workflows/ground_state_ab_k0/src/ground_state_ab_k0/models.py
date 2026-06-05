"""Interaction models used in the ground-state parameter fit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


ScalarModelFunction = Callable[[float, float, Optional[float]], Optional[float]]
DensityMap = Callable[[float, float], Optional[float]]
VolumeFractionMap = Callable[[float, float], Optional[float]]


def u_vdw(n: float, b: float, par: float | None = None) -> float:
    del b, par
    return -n


def du_vdw(n: float, b: float, par: float | None = None) -> float:
    del n, b, par
    return -1.0


def u_rks(n: float, b: float, par: float | None = None) -> float:
    del par
    if abs(b) < 1.0e-14:
        return -n
    return -(1.0 / b) * math.log(1.0 + b * n)


def du_rks(n: float, b: float, par: float | None = None) -> float:
    del par
    return -1.0 / (1.0 + b * n)


def u_pr(n: float, b: float, par: float | None = None) -> float:
    del par
    if abs(b) < 1.0e-14:
        return -n
    s2 = math.sqrt(2.0)
    num = 1.0 + (1.0 + s2) * b * n
    den = 1.0 + (1.0 - s2) * b * n
    if num <= 0.0 or den <= 0.0:
        return None
    return -(1.0 / (2.0 * s2 * b)) * math.log(num / den)


def du_pr(n: float, b: float, par: float | None = None) -> float:
    del par
    return -1.0 / (1.0 + 2.0 * b * n - (b * n) ** 2)


def u_clausius(n: float, b: float, c: float | None) -> float | None:
    del b
    if c is None:
        return None
    den = 1.0 + c * n
    if den <= 0.0:
        return None
    return -n / den


def du_clausius(n: float, b: float, c: float | None) -> float | None:
    del b
    if c is None:
        return None
    den = 1.0 + c * n
    if den <= 0.0:
        return None
    return -1.0 / (den * den)


def u_dieterici(n: float, b: float, alpha: float | None) -> float | None:
    del b
    if alpha is None or abs(alpha - 1.0) < 1.0e-14 or n <= 0.0:
        return None
    return -(n ** (alpha - 1.0)) / (alpha - 1.0)


def du_dieterici(n: float, b: float, alpha: float | None) -> float | None:
    del b
    if alpha is None or n <= 0.0:
        return None
    return -(n ** (alpha - 2.0))


def _volume_fraction_vdw(n: float, b: float) -> float | None:
    den = 1.0 - b * n
    if den <= 0.0:
        return None
    return den


def nid_from_n_ev(n: float, b: float) -> float | None:
    volume_fraction = _volume_fraction_vdw(n, b)
    if volume_fraction is None:
        return None
    return n / volume_fraction


def n_from_nid_ev(n_id: float, b: float) -> float | None:
    den = 1.0 + b * n_id
    if den <= 0.0:
        return None
    return n_id / den


def pressure_prefactor_ev(n: float, b: float) -> float | None:
    del n, b
    return 1.0


def _cs_exponent(x: float) -> float | None:
    den = 4.0 - x
    if den <= 0.0:
        return None
    return -(3.0 * x) / den - (4.0 * x) / (den * den)


def volume_fraction_cs(n: float, b: float) -> float | None:
    x = b * n
    exponent = _cs_exponent(x)
    if exponent is None:
        return None
    return math.exp(exponent)


def _volume_fraction_cs_derivative(n: float, b: float) -> float | None:
    x = b * n
    den = 4.0 - x
    if den <= 0.0:
        return None

    volume_fraction = volume_fraction_cs(n, b)
    if volume_fraction is None:
        return None

    # d/dx [ -3x/(4-x) - 4x/(4-x)^2 ] = -8(8-x)/(4-x)^3
    dg_dx = -8.0 * (8.0 - x) / (den * den * den)
    dg_dn = b * dg_dx
    return volume_fraction * dg_dn


def nid_from_n_cs(n: float, b: float) -> float | None:
    volume_fraction = volume_fraction_cs(n, b)
    if volume_fraction is None or volume_fraction <= 0.0:
        return None
    return n / volume_fraction


def pressure_prefactor_cs(n: float, b: float) -> float | None:
    volume_fraction = volume_fraction_cs(n, b)
    derivative = _volume_fraction_cs_derivative(n, b)
    if volume_fraction is None or derivative is None:
        return None
    prefactor = volume_fraction - n * derivative
    if prefactor <= 0.0:
        return None
    return prefactor


def volume_fraction_tvm(n: float, b: float) -> float:
    x = b * n
    return math.exp(-x - 0.5 * x * x)


def _volume_fraction_tvm_derivative(n: float, b: float) -> float:
    x = b * n
    volume_fraction = volume_fraction_tvm(n, b)
    return -b * (1.0 + x) * volume_fraction


def nid_from_n_tvm(n: float, b: float) -> float:
    volume_fraction = volume_fraction_tvm(n, b)
    return n / volume_fraction


def pressure_prefactor_tvm(n: float, b: float) -> float:
    volume_fraction = volume_fraction_tvm(n, b)
    derivative = _volume_fraction_tvm_derivative(n, b)
    return volume_fraction - n * derivative


def _n_from_nid_generic(
    n_id: float,
    b: float,
    *,
    volume_fraction: VolumeFractionMap,
    max_density: float | None = None,
) -> float | None:
    if n_id < 0.0:
        return None
    if n_id == 0.0:
        return 0.0
    if b <= 0.0:
        return n_id

    def target(n: float) -> float | None:
        fraction = volume_fraction(n, b)
        if fraction is None or fraction <= 0.0:
            return None
        return n / fraction - n_id

    left = 0.0
    right = min(n_id, max_density * 0.5) if max_density is not None else max(n_id, 1.0 / b)
    if right <= 0.0:
        right = 1.0 / b

    value_right = target(right)
    if value_right is None and max_density is not None:
        right = max_density * (1.0 - 1.0e-12)
        value_right = target(right)

    if value_right is None:
        return None

    attempts = 0
    while value_right < 0.0 and attempts < 200:
        if max_density is not None:
            next_right = 0.5 * (right + max_density)
            if next_right <= right:
                break
            right = next_right
        else:
            right *= 2.0

        value_right = target(right)
        if value_right is None:
            return None
        attempts += 1

    if value_right < 0.0:
        return None

    for _ in range(200):
        mid = 0.5 * (left + right)
        value_mid = target(mid)
        if value_mid is None:
            return None
        if abs(value_mid) <= 1.0e-12 * max(1.0, n_id):
            return mid
        if value_mid > 0.0:
            right = mid
        else:
            left = mid

    return 0.5 * (left + right)


def n_from_nid_cs(n_id: float, b: float) -> float | None:
    max_density = None if b <= 0.0 else 4.0 / b
    return _n_from_nid_generic(
        n_id,
        b,
        volume_fraction=volume_fraction_cs,
        max_density=max_density,
    )


def n_from_nid_tvm(n_id: float, b: float) -> float | None:
    return _n_from_nid_generic(
        n_id,
        b,
        volume_fraction=volume_fraction_tvm,
        max_density=None,
    )


@dataclass(frozen=True)
class InteractionModel:
    """Bundle the functions that define a specific interaction model."""

    name: str
    U: ScalarModelFunction
    dU: ScalarModelFunction
    nid_from_n: DensityMap
    n_from_nid: DensityMap
    volume_fraction: VolumeFractionMap
    pressure_prefactor: VolumeFractionMap


MODELS: dict[str, InteractionModel] = {
    "vdw": InteractionModel(
        "vdw",
        u_vdw,
        du_vdw,
        nid_from_n_ev,
        n_from_nid_ev,
        _volume_fraction_vdw,
        pressure_prefactor_ev,
    ),
    "rks": InteractionModel(
        "rks",
        u_rks,
        du_rks,
        nid_from_n_ev,
        n_from_nid_ev,
        _volume_fraction_vdw,
        pressure_prefactor_ev,
    ),
    "pr": InteractionModel(
        "pr",
        u_pr,
        du_pr,
        nid_from_n_ev,
        n_from_nid_ev,
        _volume_fraction_vdw,
        pressure_prefactor_ev,
    ),
    "clausius": InteractionModel(
        "clausius",
        u_clausius,
        du_clausius,
        nid_from_n_ev,
        n_from_nid_ev,
        _volume_fraction_vdw,
        pressure_prefactor_ev,
    ),
    "dieterici": InteractionModel(
        "dieterici",
        u_dieterici,
        du_dieterici,
        nid_from_n_ev,
        n_from_nid_ev,
        _volume_fraction_vdw,
        pressure_prefactor_ev,
    ),
    "cs": InteractionModel(
        "cs",
        u_vdw,
        du_vdw,
        nid_from_n_cs,
        n_from_nid_cs,
        volume_fraction_cs,
        pressure_prefactor_cs,
    ),
    "tvm": InteractionModel(
        "tvm",
        u_vdw,
        du_vdw,
        nid_from_n_tvm,
        n_from_nid_tvm,
        volume_fraction_tvm,
        pressure_prefactor_tvm,
    ),
}
