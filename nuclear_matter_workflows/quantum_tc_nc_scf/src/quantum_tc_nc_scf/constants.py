"""Constants and numerical settings for the quantum SCF critical-point solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def _make_grid(start: float, stop: float, count: int) -> Tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (float(start),)

    step = (stop - start) / float(count - 1)
    return tuple(start + step * float(i) for i in range(count))


@dataclass(frozen=True)
class QuantumCriticalSettings:
    """Physical constants and solver settings for the finite-`T` problem."""

    hbarc: float = 197.3269804
    m_nuc: float = 938.0
    dgen: float = 4.0
    n0: float = 0.16

    n_k_fd: int = 400
    k_max_fd: float = 20.0

    mu_min: float = -200.0
    mu_max: float = 1200.0
    mu_scf_tol: float = 1.0e-8
    mu_scf_max_iter: int = 80
    mu_scf_damping: float = 0.70
    mu_delta: float = 0.25

    t_min_cp: float = 10.0
    t_max_cp: float = 22.0
    n_min_cp: float = 0.045
    n_max_cp: float = 0.075

    coarse_t_count: int = 17
    coarse_n_count: int = 17

    h_n: float = 2.0e-4
    h_t: float = 2.0e-2

    outer_tol_dPdn: float = 1.0e-5
    outer_tol_d2Pdn2: float = 1.0e-4
    outer_max_iter: int = 35
    outer_damping_t: float = 0.70
    outer_damping_n: float = 0.70
    outer_max_step_t: float = 0.75
    outer_max_step_n: float = 0.004
    outer_min_lambda: float = 1.0e-3

    cache_round_digits: int = 8

    @staticmethod
    def quick() -> "QuantumCriticalSettings":
        """Reduced settings for faster interactive checks."""
        return QuantumCriticalSettings(
            n_k_fd=220,
            coarse_t_count=11,
            coarse_n_count=11,
            outer_max_iter=24,
        )


DEFAULT_SETTINGS = QuantumCriticalSettings()
DEFAULT_ALPHA_LIST = _make_grid(5.0 / 3.0, 2.0, 10)
DEFAULT_C_LIST = _make_grid(0.0, 4.74, 10)

