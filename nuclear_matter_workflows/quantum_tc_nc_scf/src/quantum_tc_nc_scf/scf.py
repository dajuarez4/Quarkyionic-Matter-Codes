"""Self-consistent-field routines for the quantum critical-point problem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ground_state_ab_k0.models import MODELS

from .constants import DEFAULT_SETTINGS, QuantumCriticalSettings
from .quantum_fermi import QuantumFermiGas
from .utils.numerics import linspace


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ScfCriticalPoint:
    """Quantum critical point obtained from the SCF solver."""

    Tc: float
    nc: float
    Pc: float
    dPdn: float
    d2Pdn2: float
    score: float
    iterations: int
    mu_star: float


class QuantumPressureEvaluator:
    """Evaluate the quantum pressure `P(T, n)` by inverting to `mu*` with SCF."""

    def __init__(
        self,
        model_name: str,
        a: float,
        b: float,
        par: Optional[float],
        settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    ) -> None:
        self.model_name = model_name
        self.a = a
        self.b = b
        self.par = par
        self.settings = settings
        self.model = MODELS[model_name]
        self.fermi = QuantumFermiGas(settings)
        self._pressure_cache: Dict[Tuple[float, float], Tuple[Optional[float], Optional[float]]] = {}

    def _cache_key(self, T: float, n: float) -> Tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(T), digits), round(float(n), digits))

    def target_n_id(self, n: float) -> Optional[float]:
        return self.model.nid_from_n(n, self.b)

    def solve_mu_star_scf(self, T: float, n: float, mu0: Optional[float] = None) -> Optional[float]:
        """Solve `n_id(T, mu*) = n_id_target` by damped SCF iteration."""
        target_n_id = self.target_n_id(n)
        if target_n_id is None or target_n_id <= 0.0 or T <= 0.0:
            return None

        settings = self.settings
        mu = self.fermi.mu_seed_from_n_id(target_n_id) if mu0 is None else mu0
        mu = _clamp(mu, settings.mu_min, settings.mu_max)

        for _ in range(settings.mu_scf_max_iter):
            n_id_value = self.fermi.number_density(T, mu)
            residual = n_id_value - target_n_id

            if abs(residual) <= settings.mu_scf_tol * max(1.0, target_n_id):
                return mu

            mu_plus = min(settings.mu_max, mu + settings.mu_delta)
            mu_minus = max(settings.mu_min, mu - settings.mu_delta)
            if mu_plus <= mu_minus:
                return None

            n_plus = self.fermi.number_density(T, mu_plus)
            n_minus = self.fermi.number_density(T, mu_minus)
            susceptibility = (n_plus - n_minus) / (mu_plus - mu_minus)

            if not math.isfinite(susceptibility) or abs(susceptibility) < 1.0e-12:
                return None

            update = residual / susceptibility
            mu_new = mu - settings.mu_scf_damping * update
            mu_new = _clamp(mu_new, settings.mu_min, settings.mu_max)

            if abs(mu_new - mu) <= settings.mu_scf_tol * max(1.0, abs(mu)):
                mu = mu_new
                if abs(residual) <= 10.0 * settings.mu_scf_tol * max(1.0, target_n_id):
                    return mu

            mu = mu_new

        return None

    def pressure(self, T: float, n: float) -> Tuple[Optional[float], Optional[float]]:
        """Return `(P, mu*)` for a given `T` and physical density `n`."""
        key = self._cache_key(T, n)
        if key in self._pressure_cache:
            return self._pressure_cache[key]

        mu_star = self.solve_mu_star_scf(T, n)
        if mu_star is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        _, p_id = self.fermi.evaluate_number_and_pressure(T, mu_star)
        pressure_prefactor = self.model.pressure_prefactor(n, self.b)
        if pressure_prefactor is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        dU_value = self.model.dU(n, self.b, self.par)
        if dU_value is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        pressure_value = pressure_prefactor * p_id + n * n * self.a * dU_value
        result = (pressure_value, mu_star)
        self._pressure_cache[key] = result
        return result

    def critical_equations(self, T: float, n: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return `(dP/dn, d2P/dn2, mu*)` at a given `(T, n)`."""
        settings = self.settings
        h_n = settings.h_n

        n_minus = n - h_n
        n_plus = n + h_n
        if n_minus <= settings.n_min_cp or n_plus >= settings.n_max_cp:
            return None, None, None

        P_minus, _ = self.pressure(T, n_minus)
        P_zero, mu_star = self.pressure(T, n)
        P_plus, _ = self.pressure(T, n_plus)

        if P_minus is None or P_zero is None or P_plus is None:
            return None, None, None

        dPdn = (P_plus - P_minus) / (2.0 * h_n)
        d2Pdn2 = (P_plus - 2.0 * P_zero + P_minus) / (h_n * h_n)
        return dPdn, d2Pdn2, mu_star


def _build_seed_grid(settings: QuantumCriticalSettings) -> Tuple[List[float], List[float]]:
    t_grid = linspace(settings.t_min_cp, settings.t_max_cp, settings.coarse_t_count)
    n_grid = linspace(settings.n_min_cp, settings.n_max_cp, settings.coarse_n_count)
    return t_grid, n_grid


def find_seed_point(
    evaluator: QuantumPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """Return a coarse seed and normalization scales for the SCF iteration."""
    t_grid, n_grid = _build_seed_grid(settings)
    candidates: List[Tuple[float, float, float, float]] = []

    if preferred_seed is not None:
        T_seed, n_seed = preferred_seed
        values = evaluator.critical_equations(T_seed, n_seed)
        if values[0] is not None and values[1] is not None:
            candidates.append((T_seed, n_seed, float(values[0]), float(values[1])))

    for T in t_grid:
        for n in n_grid:
            dPdn, d2Pdn2, _ = evaluator.critical_equations(T, n)
            if dPdn is None or d2Pdn2 is None:
                continue
            if not math.isfinite(dPdn) or not math.isfinite(d2Pdn2):
                continue
            candidates.append((T, n, dPdn, d2Pdn2))

    if not candidates:
        return None

    scale_1 = max(abs(item[2]) for item in candidates)
    scale_2 = max(abs(item[3]) for item in candidates)
    scale_1 = max(scale_1, 1.0e-12)
    scale_2 = max(scale_2, 1.0e-12)

    best = None
    for T, n, dPdn, d2Pdn2 in candidates:
        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if best is None or score < best[0]:
            best = (score, T, n)

    if best is None:
        return None

    return (best[1], best[2]), scale_1, scale_2


def solve_critical_point_scf(
    evaluator: QuantumPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[ScfCriticalPoint]:
    """Solve the quantum critical-point equations with damped SCF updates."""
    seed_info = find_seed_point(evaluator, settings=settings, preferred_seed=preferred_seed)
    if seed_info is None:
        return None

    (T, n), scale_1, scale_2 = seed_info
    score = float("inf")

    for iteration in range(1, settings.outer_max_iter + 1):
        dPdn, d2Pdn2, mu_star = evaluator.critical_equations(T, n)
        if dPdn is None or d2Pdn2 is None or mu_star is None:
            return None

        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if abs(dPdn) <= settings.outer_tol_dPdn and abs(d2Pdn2) <= settings.outer_tol_d2Pdn2:
            pressure_value, mu_star_final = evaluator.pressure(T, n)
            if pressure_value is None or mu_star_final is None:
                return None
            return ScfCriticalPoint(
                Tc=T,
                nc=n,
                Pc=pressure_value,
                dPdn=dPdn,
                d2Pdn2=d2Pdn2,
                score=score,
                iterations=iteration,
                mu_star=mu_star_final,
            )

        T_plus = min(settings.t_max_cp, T + settings.h_t)
        T_minus = max(settings.t_min_cp, T - settings.h_t)
        if T_plus <= T_minus:
            return None

        dPdn_plus, _, _ = evaluator.critical_equations(T_plus, n)
        dPdn_minus, _, _ = evaluator.critical_equations(T_minus, n)
        if dPdn_plus is None or dPdn_minus is None:
            return None
        slope_T = (dPdn_plus - dPdn_minus) / (T_plus - T_minus)

        n_plus = min(settings.n_max_cp, n + settings.h_n)
        n_minus = max(settings.n_min_cp, n - settings.h_n)
        if n_plus <= n_minus:
            return None

        _, d2Pdn2_plus, _ = evaluator.critical_equations(T, n_plus)
        _, d2Pdn2_minus, _ = evaluator.critical_equations(T, n_minus)
        if d2Pdn2_plus is None or d2Pdn2_minus is None:
            return None
        slope_n = (d2Pdn2_plus - d2Pdn2_minus) / (n_plus - n_minus)

        if abs(slope_T) < 1.0e-12 or abs(slope_n) < 1.0e-12:
            return None

        delta_T = -settings.outer_damping_t * dPdn / slope_T
        delta_n = -settings.outer_damping_n * d2Pdn2 / slope_n

        delta_T = _clamp(delta_T, -settings.outer_max_step_t, settings.outer_max_step_t)
        delta_n = _clamp(delta_n, -settings.outer_max_step_n, settings.outer_max_step_n)

        accepted = False
        lam = 1.0
        while lam >= settings.outer_min_lambda:
            T_try = _clamp(T + lam * delta_T, settings.t_min_cp, settings.t_max_cp)
            n_try = _clamp(n + lam * delta_n, settings.n_min_cp, settings.n_max_cp)

            dPdn_try, d2Pdn2_try, _ = evaluator.critical_equations(T_try, n_try)
            if dPdn_try is None or d2Pdn2_try is None:
                lam *= 0.5
                continue

            score_try = (dPdn_try / scale_1) ** 2 + (d2Pdn2_try / scale_2) ** 2
            if score_try < score:
                T = T_try
                n = n_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return None

    pressure_value, mu_star_final = evaluator.pressure(T, n)
    if pressure_value is None or mu_star_final is None:
        return None

    dPdn, d2Pdn2, _ = evaluator.critical_equations(T, n)
    if dPdn is None or d2Pdn2 is None:
        return None

    score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
    return ScfCriticalPoint(
        Tc=T,
        nc=n,
        Pc=pressure_value,
        dPdn=dPdn,
        d2Pdn2=d2Pdn2,
        score=score,
        iterations=settings.outer_max_iter,
        mu_star=mu_star_final,
    )
