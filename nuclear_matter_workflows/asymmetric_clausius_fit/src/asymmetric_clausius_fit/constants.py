"""Constants and default target grids for the asymmetric Clausius workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsymmetricClausiusConstants:
    """Physical and numerical constants for the zero-temperature fit."""

    hbarc: float = 197.3269804
    m_nucleon: float = 938.0

    symmetric_degeneracy: float = 4.0
    species_degeneracy: float = 2.0

    n0: float = 0.16
    binding_energy: float = -16.0
    symmetry_energy_j: float = 32.5
    symmetry_slope_l: float = 58.9

    b_min: float = 1.0e-6
    b_max: float = 6.0
    b_scan_steps: int = 4000

    c_min: float = 0.0
    c_max: float = 4.74
    c_scan_steps: int = 240

    k0_derivative_step: float = 1.0e-6
    l_derivative_step: float = 1.0e-4

    saturation_left: float = 0.05
    saturation_right: float = 0.30


DEFAULT_CONSTANTS = AsymmetricClausiusConstants()

DEFAULT_K0_TARGETS: tuple[float, ...] = (
    250.0,
    260.0,
    270.0,
    280.0,
    290.0,
    300.0,
    310.0,
    315.0,
)
