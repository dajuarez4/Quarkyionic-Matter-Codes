"""Finite-temperature fixed-composition Clausius workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

from quantum_tc_nc_scf.constants import DEFAULT_SETTINGS, QuantumCriticalSettings
from quantum_tc_nc_scf.quantum_fermi import QuantumFermiGas
from quantum_tc_nc_scf.utils.numerics import linspace


@dataclass(frozen=True)
class FixedYCriticalPoint:
    """Critical point for one fixed proton fraction `y`."""

    target_k0: float
    y: float
    Tc: float
    nc: float
    Pc: float
    dPdn: float
    d2Pdn2: float
    score: float
    iterations: int
    mu_p_star: float
    mu_n_star: float
    c: float
    a_avg: float
    b_avg: float
    a_n: float
    a_pn: float
    b_n: float
    b_pn: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _attractive_coefficient(y: float, a_n: float, a_pn: float) -> float:
    return a_n * (y * y + (1.0 - y) ** 2) + 2.0 * a_pn * y * (1.0 - y)


def _species_data(
    n: float,
    y: float,
    b_n: float,
    b_pn: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    if n <= 0.0 or y < 0.0 or y > 0.5:
        return None

    n_p = n * y
    n_n = n * (1.0 - y)

    x_p = b_n * n_p + b_pn * n_n
    x_n = b_pn * n_p + b_n * n_n
    f_p = 1.0 - x_p
    f_n = 1.0 - x_n
    if f_p <= 0.0 or f_n <= 0.0:
        return None

    n_p_id = 0.0 if n_p <= 0.0 else n_p / f_p
    n_n_id = 0.0 if n_n <= 0.0 else n_n / f_n
    return n_p, n_n, f_p, f_n, n_p_id, n_n_id


class FixedYAsymmetricClausiusPressureEvaluator:
    """Evaluate `P(T,n;y)` for asymmetric Clausius matter at fixed composition."""

    def __init__(
        self,
        *,
        a_n: float,
        a_pn: float,
        b_n: float,
        b_pn: float,
        c: float,
        y: float,
        settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    ) -> None:
        self.a_n = float(a_n)
        self.a_pn = float(a_pn)
        self.b_n = float(b_n)
        self.b_pn = float(b_pn)
        self.c = float(c)
        self.y = float(y)
        self.settings = settings

        species_settings = replace(settings, dgen=2.0)
        self.proton_fermi = QuantumFermiGas(species_settings)
        self.neutron_fermi = QuantumFermiGas(species_settings)
        self._pressure_cache: Dict[Tuple[float, float], Tuple[Optional[float], Optional[float], Optional[float]]] = {}

    def _cache_key(self, T: float, n: float) -> Tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(T), digits), round(float(n), digits))

    def target_n_id(self, n: float) -> Optional[Tuple[float, float]]:
        data = _species_data(n, self.y, self.b_n, self.b_pn)
        if data is None:
            return None
        return data[4], data[5]

    def _solve_mu_star_species(
        self,
        fermi: QuantumFermiGas,
        T: float,
        n_id_target: float,
        mu0: Optional[float] = None,
    ) -> Optional[float]:
        if T <= 0.0 or n_id_target < 0.0:
            return None
        if n_id_target == 0.0:
            return self.settings.m_nuc

        mu = fermi.mu_seed_from_n_id(n_id_target) if mu0 is None else mu0
        mu = _clamp(mu, self.settings.mu_min, self.settings.mu_max)

        for _ in range(self.settings.mu_scf_max_iter):
            n_id_value = fermi.number_density(T, mu)
            residual = n_id_value - n_id_target
            if abs(residual) <= self.settings.mu_scf_tol * max(1.0, n_id_target):
                return mu

            mu_plus = min(self.settings.mu_max, mu + self.settings.mu_delta)
            mu_minus = max(self.settings.mu_min, mu - self.settings.mu_delta)
            if mu_plus <= mu_minus:
                return None

            n_plus = fermi.number_density(T, mu_plus)
            n_minus = fermi.number_density(T, mu_minus)
            susceptibility = (n_plus - n_minus) / (mu_plus - mu_minus)
            if not math.isfinite(susceptibility) or abs(susceptibility) < 1.0e-12:
                return None

            update = residual / susceptibility
            mu_new = mu - self.settings.mu_scf_damping * update
            mu_new = _clamp(mu_new, self.settings.mu_min, self.settings.mu_max)

            if abs(mu_new - mu) <= self.settings.mu_scf_tol * max(1.0, abs(mu)):
                mu = mu_new
                if abs(residual) <= 10.0 * self.settings.mu_scf_tol * max(1.0, n_id_target):
                    return mu

            mu = mu_new

        return None

    def pressure(self, T: float, n: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return `(P, mu_p*, mu_n*)` for one fixed-`y` branch."""
        key = self._cache_key(T, n)
        if key in self._pressure_cache:
            return self._pressure_cache[key]

        targets = self.target_n_id(n)
        if targets is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None
        n_p_id_target, n_n_id_target = targets

        mu_p = self._solve_mu_star_species(self.proton_fermi, T, n_p_id_target)
        mu_n = self._solve_mu_star_species(self.neutron_fermi, T, n_n_id_target)
        if mu_p is None or mu_n is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None

        p_p = self.proton_fermi.pressure(T, mu_p)
        p_n = self.neutron_fermi.pressure(T, mu_n)

        coeff = _attractive_coefficient(self.y, self.a_n, self.a_pn)
        den = 1.0 + self.c * n
        if den <= 0.0:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None

        p_int = -coeff * n * n / (den * den)
        result = (p_p + p_n + p_int, mu_p, mu_n)
        self._pressure_cache[key] = result
        return result

    def critical_equations(self, T: float, n: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return `(dP/dn, d2P/dn2, mu_p*, mu_n*)` for one fixed-`y` branch."""
        h_n = self.settings.h_n
        n_minus = n - h_n
        n_plus = n + h_n
        if n_minus <= self.settings.n_min_cp or n_plus >= self.settings.n_max_cp:
            return None, None, None, None

        p_minus, _, _ = self.pressure(T, n_minus)
        p_zero, mu_p, mu_n = self.pressure(T, n)
        p_plus, _, _ = self.pressure(T, n_plus)
        if p_minus is None or p_zero is None or p_plus is None or mu_p is None or mu_n is None:
            return None, None, None, None

        dPdn = (p_plus - p_minus) / (2.0 * h_n)
        d2Pdn2 = (p_plus - 2.0 * p_zero + p_minus) / (h_n * h_n)
        return dPdn, d2Pdn2, mu_p, mu_n


def _build_seed_grid(settings: QuantumCriticalSettings) -> Tuple[list[float], list[float]]:
    return (
        linspace(settings.t_min_cp, settings.t_max_cp, settings.coarse_t_count),
        linspace(settings.n_min_cp, settings.n_max_cp, settings.coarse_n_count),
    )


def find_seed_point_fixed_y(
    evaluator: FixedYAsymmetricClausiusPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """Return a coarse critical seed and normalization scales."""
    t_grid, n_grid = _build_seed_grid(settings)
    candidates: list[Tuple[float, float, float, float]] = []

    if preferred_seed is not None:
        T_seed, n_seed = preferred_seed
        values = evaluator.critical_equations(T_seed, n_seed)
        if values[0] is not None and values[1] is not None:
            candidates.append((T_seed, n_seed, float(values[0]), float(values[1])))

    for T in t_grid:
        for n in n_grid:
            dPdn, d2Pdn2, _, _ = evaluator.critical_equations(T, n)
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


def solve_critical_point_fixed_y_scf(
    evaluator: FixedYAsymmetricClausiusPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float, float, float, float, float, int, float, float]]:
    """Solve the fixed-`y` critical-point equations with damped SCF updates."""
    seed_info = find_seed_point_fixed_y(evaluator, settings=settings, preferred_seed=preferred_seed)
    if seed_info is None:
        return None

    (T, n), scale_1, scale_2 = seed_info
    score = float("inf")

    for iteration in range(1, settings.outer_max_iter + 1):
        dPdn, d2Pdn2, mu_p, mu_n = evaluator.critical_equations(T, n)
        if dPdn is None or d2Pdn2 is None or mu_p is None or mu_n is None:
            return None

        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if abs(dPdn) <= settings.outer_tol_dPdn and abs(d2Pdn2) <= settings.outer_tol_d2Pdn2:
            pressure_value, mu_p_final, mu_n_final = evaluator.pressure(T, n)
            if pressure_value is None or mu_p_final is None or mu_n_final is None:
                return None
            return (
                T,
                n,
                pressure_value,
                dPdn,
                d2Pdn2,
                score,
                iteration,
                mu_p_final,
                mu_n_final,
            )

        T_plus = min(settings.t_max_cp, T + settings.h_t)
        T_minus = max(settings.t_min_cp, T - settings.h_t)
        if T_plus <= T_minus:
            return None

        dPdn_plus, _, _, _ = evaluator.critical_equations(T_plus, n)
        dPdn_minus, _, _, _ = evaluator.critical_equations(T_minus, n)
        if dPdn_plus is None or dPdn_minus is None:
            return None
        slope_T = (dPdn_plus - dPdn_minus) / (T_plus - T_minus)

        n_plus = min(settings.n_max_cp, n + settings.h_n)
        n_minus = max(settings.n_min_cp, n - settings.h_n)
        if n_plus <= n_minus:
            return None

        _, d2Pdn2_plus, _, _ = evaluator.critical_equations(T, n_plus)
        _, d2Pdn2_minus, _, _ = evaluator.critical_equations(T, n_minus)
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

            dPdn_try, d2Pdn2_try, _, _ = evaluator.critical_equations(T_try, n_try)
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

    pressure_value, mu_p_final, mu_n_final = evaluator.pressure(T, n)
    if pressure_value is None or mu_p_final is None or mu_n_final is None:
        return None

    dPdn, d2Pdn2, _, _ = evaluator.critical_equations(T, n)
    if dPdn is None or d2Pdn2 is None:
        return None

    score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
    return (
        T,
        n,
        pressure_value,
        dPdn,
        d2Pdn2,
        score,
        settings.outer_max_iter,
        mu_p_final,
        mu_n_final,
    )


def compute_fixed_y_critical_point_from_fit_row(
    row,
    y: float,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[FixedYCriticalPoint]:
    """Compute one fixed-`y` finite-temperature critical point from a fit-table row."""
    evaluator = FixedYAsymmetricClausiusPressureEvaluator(
        a_n=float(row.a_n),
        a_pn=float(row.a_pn),
        b_n=float(row.b_n),
        b_pn=float(row.b_pn),
        c=float(row.c),
        y=float(y),
        settings=settings,
    )
    solved = solve_critical_point_fixed_y_scf(evaluator, settings=settings, preferred_seed=preferred_seed)
    if solved is None:
        return None

    Tc, nc, Pc, dPdn, d2Pdn2, score, iterations, mu_p_star, mu_n_star = solved
    return FixedYCriticalPoint(
        target_k0=float(row.target_k0),
        y=float(y),
        Tc=Tc,
        nc=nc,
        Pc=Pc,
        dPdn=dPdn,
        d2Pdn2=d2Pdn2,
        score=score,
        iterations=iterations,
        mu_p_star=mu_p_star,
        mu_n_star=mu_n_star,
        c=float(row.c),
        a_avg=float(row.a_avg),
        b_avg=float(row.b_avg),
        a_n=float(row.a_n),
        a_pn=float(row.a_pn),
        b_n=float(row.b_n),
        b_pn=float(row.b_pn),
    )


def compute_fixed_y_family_from_fit_df(
    fit_df,
    y_values: Iterable[float],
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
) -> list[FixedYCriticalPoint]:
    """Compute a fixed-`y` finite-temperature family using continuation seeds."""
    results: list[FixedYCriticalPoint] = []
    for row in fit_df.itertuples(index=False):
        previous_seed: Optional[Tuple[float, float]] = None
        for y in y_values:
            result = compute_fixed_y_critical_point_from_fit_row(
                row,
                float(y),
                settings=settings,
                preferred_seed=previous_seed,
            )
            if result is not None:
                results.append(result)
                previous_seed = (result.Tc, result.nc)
            else:
                previous_seed = None
    return results
