"""Finite-temperature ideal Fermi-gas integrals."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .constants import DEFAULT_SETTINGS, QuantumCriticalSettings
from .utils.numerics import simpson_weights


class QuantumFermiGas:
    """Evaluate finite-temperature ideal-gas densities and pressures."""

    def __init__(self, settings: QuantumCriticalSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings
        self.k_grid, self.w_grid = simpson_weights(0.0, settings.k_max_fd, settings.n_k_fd)
        self.e_grid = [
            math.sqrt((settings.hbarc * k) ** 2 + settings.m_nuc ** 2) for k in self.k_grid
        ]
        self.k2_grid = [k * k for k in self.k_grid]
        self.pk_grid = [
            (settings.hbarc ** 2) * (k ** 4) / e for k, e in zip(self.k_grid, self.e_grid)
        ]
        self.pref_n = settings.dgen / (2.0 * math.pi ** 2)
        self.pref_p = settings.dgen / (6.0 * math.pi ** 2)
        self._cache: Dict[Tuple[float, float], Tuple[float, float]] = {}

    def _cache_key(self, T: float, mu: float) -> Tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(T), digits), round(float(mu), digits))

    @staticmethod
    def fermi_dirac(E: float, mu: float, T: float) -> float:
        """Fermi-Dirac occupation factor with overflow protection."""
        arg = (E - mu) / T
        if arg > 700.0:
            return 0.0
        if arg < -700.0:
            return 1.0
        return 1.0 / (math.exp(arg) + 1.0)

    def evaluate_number_and_pressure(self, T: float, mu: float) -> Tuple[float, float]:
        """Return `(n_id, p_id)` at finite `T` and effective chemical potential `mu*`."""
        key = self._cache_key(T, mu)
        if key in self._cache:
            return self._cache[key]

        s_n = 0.0
        s_p = 0.0
        for weight, E, k2, pk in zip(self.w_grid, self.e_grid, self.k2_grid, self.pk_grid):
            f = self.fermi_dirac(E, mu, T)
            s_n += weight * k2 * f
            s_p += weight * pk * f

        result = (self.pref_n * s_n, self.pref_p * s_p)
        self._cache[key] = result
        return result

    def number_density(self, T: float, mu: float) -> float:
        """Return the finite-temperature ideal-gas density."""
        return self.evaluate_number_and_pressure(T, mu)[0]

    def pressure(self, T: float, mu: float) -> float:
        """Return the finite-temperature ideal-gas pressure."""
        return self.evaluate_number_and_pressure(T, mu)[1]

    def mu_seed_from_n_id(self, n_id_target: float) -> float:
        """Zero-temperature estimate used as an initial guess for `mu*`."""
        if n_id_target <= 0.0:
            return self.settings.m_nuc

        kf = (6.0 * math.pi ** 2 * n_id_target / self.settings.dgen) ** (1.0 / 3.0)
        return math.sqrt((self.settings.hbarc * kf) ** 2 + self.settings.m_nuc ** 2)

