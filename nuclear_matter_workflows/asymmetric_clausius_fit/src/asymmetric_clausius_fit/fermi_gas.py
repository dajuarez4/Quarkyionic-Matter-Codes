"""Analytic zero-temperature Fermi-gas relations in fm/MeV units."""

from __future__ import annotations

import math

from .constants import DEFAULT_CONSTANTS, AsymmetricClausiusConstants


def kf_from_nid(
    n_id: float,
    degeneracy: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    """Return the Fermi momentum for an ideal-gas density."""
    del constants
    if n_id <= 0.0:
        return 0.0
    return (6.0 * math.pi ** 2 * n_id / degeneracy) ** (1.0 / 3.0)


def eps_id_from_nid(
    n_id: float,
    degeneracy: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    """Return the relativistic ideal-gas energy density."""
    if n_id <= 0.0:
        return 0.0

    kf = kf_from_nid(n_id, degeneracy, constants)
    pf = constants.hbarc * kf
    ef = math.sqrt(pf * pf + constants.m_nucleon ** 2)
    term = (
        pf * ef * (2.0 * pf * pf + constants.m_nucleon ** 2)
        - constants.m_nucleon ** 4 * math.log((pf + ef) / constants.m_nucleon)
    )
    return degeneracy * term / (16.0 * math.pi ** 2 * constants.hbarc ** 3)


def p_id_from_nid(
    n_id: float,
    degeneracy: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    """Return the relativistic ideal-gas pressure."""
    if n_id <= 0.0:
        return 0.0

    kf = kf_from_nid(n_id, degeneracy, constants)
    pf = constants.hbarc * kf
    ef = math.sqrt(pf * pf + constants.m_nucleon ** 2)
    term = (
        pf * ef * (2.0 * pf * pf - 3.0 * constants.m_nucleon ** 2)
        + 3.0
        * constants.m_nucleon ** 4
        * math.log((pf + ef) / constants.m_nucleon)
    )
    return degeneracy * term / (48.0 * math.pi ** 2 * constants.hbarc ** 3)
