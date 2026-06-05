"""Ideal Fermi-gas relations at zero temperature."""

from __future__ import annotations

import math

from .constants import DEFAULT_CONSTANTS, GroundStateConstants
from .utils.numerics import simpson_integral


def energy_dispersion(k: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Single-particle relativistic dispersion relation."""
    return math.sqrt((constants.hbarc * k) ** 2 + constants.m_nuc ** 2)


def kf_from_nid(n_id: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Fermi momentum from ideal-gas density."""
    if n_id <= 0.0:
        return 0.0
    return (6.0 * math.pi ** 2 * n_id / constants.dgen) ** (1.0 / 3.0)


def n_id_from_kf(kf: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Ideal-gas density from Fermi momentum."""
    return constants.dgen * kf ** 3 / (6.0 * math.pi ** 2)


def eps_id_from_kf(kf: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Ideal-gas energy density from Fermi momentum."""
    if kf <= 0.0:
        return 0.0

    def integrand(k: float) -> float:
        return k * k * energy_dispersion(k, constants)

    integral = simpson_integral(integrand, 0.0, kf, constants.n_int)
    return constants.dgen * integral / (2.0 * math.pi ** 2)


def p_id_from_kf(kf: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Ideal-gas pressure from Fermi momentum."""
    if kf <= 0.0:
        return 0.0

    def integrand(k: float) -> float:
        ek = energy_dispersion(k, constants)
        return (constants.hbarc ** 2) * k ** 4 / ek

    integral = simpson_integral(integrand, 0.0, kf, constants.n_int)
    return constants.dgen * integral / (6.0 * math.pi ** 2)


def eps_id_from_nid(n_id: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Ideal-gas energy density from ideal-gas density."""
    return eps_id_from_kf(kf_from_nid(n_id, constants), constants)


def p_id_from_nid(n_id: float, constants: GroundStateConstants = DEFAULT_CONSTANTS) -> float:
    """Ideal-gas pressure from ideal-gas density."""
    return p_id_from_kf(kf_from_nid(n_id, constants), constants)

