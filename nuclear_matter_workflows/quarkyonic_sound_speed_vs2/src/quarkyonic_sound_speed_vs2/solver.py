"""Quarkyonic sound-speed solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ground_state_ab_k0 import compute_ground_state_point
from ground_state_ab_k0.constants import DEFAULT_CONSTANTS
from ground_state_ab_k0.models import MODELS

from .constants import DEFAULT_SETTINGS, QuarkyonicSettings
from .utils.numerics import (
    first_derivative_from_grid,
    golden_section_min,
    linspace,
    second_derivative_from_grid,
    simpson_integral,
    smooth_local_polynomial,
)


@dataclass(frozen=True)
class QuarkyonicCurve:
    """One quarkyonic sound-speed curve together with the fitted hadronic parameters."""

    model: str
    par: Optional[float]
    a: float
    b: float
    K0: float
    n: list[float]
    n_over_n0: list[float]
    eps: list[float]
    mu_b: list[float]
    P: list[float]
    dP_dn: list[float]
    d2eps_dn2: list[float]
    vs2: list[float]
    quark_fraction: list[float]
    n_q: list[float]
    n_n: list[float]
    k_bu: list[float]
    k_f: list[float]
    vs2_max: float
    n_tr: float
    n_tr_over_n0: float


@dataclass(frozen=True)
class QuarkyonicState:
    """Minimum-energy state at one fixed baryon density."""

    n_b: float
    n_over_n0: float
    quark_fraction: float
    n_q: float
    n_n: float
    k_bu: float
    k_f: float
    energy_density: float


def _nucleon_energy_shell(
    k_bu: float,
    k_f: float,
    shell_points: int,
) -> float:
    """Return the ideal nucleon shell energy density between `k_bu` and `k_f`."""
    if k_f <= k_bu:
        return 0.0

    def integrand(k: float) -> float:
        energy = math.sqrt((DEFAULT_CONSTANTS.hbarc * k) ** 2 + DEFAULT_CONSTANTS.m_nuc ** 2)
        return k * k * energy

    integral = simpson_integral(integrand, k_bu, k_f, shell_points)
    return DEFAULT_CONSTANTS.dgen * integral / (2.0 * math.pi ** 2)


def _quark_density_from_kbu(k_bu: float, settings: QuarkyonicSettings) -> float:
    """Return the baryon density carried by quarks for a given `k_bu`."""
    if k_bu <= 0.0:
        return 0.0

    upper = k_bu / float(settings.nc)
    kappa = settings.lambda_momentum_mev / DEFAULT_CONSTANTS.hbarc

    def integrand(q: float) -> float:
        return q * math.sqrt(kappa * kappa + q * q)

    integral = simpson_integral(integrand, 0.0, upper, settings.quark_integral_points)
    return settings.quark_degeneracy * integral / (2.0 * math.pi ** 2)


def _quark_energy_density_from_kbu(k_bu: float, settings: QuarkyonicSettings) -> float:
    """Return the quark energy density for a given `k_bu`."""
    if k_bu <= 0.0:
        return 0.0

    upper = k_bu / float(settings.nc)
    kappa = settings.lambda_momentum_mev / DEFAULT_CONSTANTS.hbarc
    m_q = DEFAULT_CONSTANTS.m_nuc / float(settings.nc)

    def integrand(q: float) -> float:
        density_factor = q * math.sqrt(kappa * kappa + q * q)
        energy = math.sqrt((DEFAULT_CONSTANTS.hbarc * q) ** 2 + m_q * m_q)
        return density_factor * energy

    integral = simpson_integral(integrand, 0.0, upper, settings.quark_integral_points)
    prefactor = float(settings.nc) * settings.quark_degeneracy / (2.0 * math.pi ** 2)
    return prefactor * integral


def _kbu_from_quark_density(n_q: float, settings: QuarkyonicSettings) -> float:
    """Invert the quark density relation analytically to obtain `k_bu`."""
    if n_q <= 0.0:
        return 0.0

    kappa = settings.lambda_momentum_mev / DEFAULT_CONSTANTS.hbarc
    term = kappa ** 3 + (6.0 * math.pi ** 2 * n_q) / settings.quark_degeneracy
    val = term ** (2.0 / 3.0) - kappa * kappa
    return float(settings.nc) * math.sqrt(max(val, 0.0))


def _lower_quark_fraction_bound(n_b: float, b: float, settings: QuarkyonicSettings) -> float:
    """Return the minimum allowed quark fraction at baryon density `n_b`."""
    if n_b <= 0.0 or b <= 0.0:
        return 0.0
    return max(0.0, 1.0 - 1.0 / (b * n_b) + settings.fq_min_shift)


def _state_at_fraction(
    model_name: str,
    a: float,
    b: float,
    n_b: float,
    fq: float,
    settings: QuarkyonicSettings,
    par: Optional[float] = None,
) -> Optional[QuarkyonicState]:
    """Build the hadron+quark state at fixed `n_b` and quark fraction `fq`."""
    model = MODELS[model_name]

    if fq < 0.0 or fq > 1.0:
        return None

    n_q = n_b * fq
    n_n = n_b - n_q
    if n_n < 0.0:
        return None

    k_bu = _kbu_from_quark_density(n_q, settings)

    if n_n <= 1.0e-14:
        n_n_id = 0.0
        k_f = k_bu
        shell_energy = 0.0
        interaction_energy = 0.0
    else:
        n_n_id = model.nid_from_n(n_n, b)
        if n_n_id is None:
            return None

        k_f = (k_bu ** 3 + (6.0 * math.pi ** 2 * n_n_id) / DEFAULT_CONSTANTS.dgen) ** (1.0 / 3.0)
        shell_energy = _nucleon_energy_shell(k_bu, k_f, settings.shell_integral_points)

        volume_fraction = n_n / n_n_id if n_n_id > 0.0 else 1.0
        u_value = model.U(n_n, b, par)
        if u_value is None:
            return None
        interaction_energy = n_n * a * u_value
        shell_energy = volume_fraction * shell_energy

    quark_energy = _quark_energy_density_from_kbu(k_bu, settings)
    total_energy = shell_energy + interaction_energy + quark_energy

    if not math.isfinite(total_energy):
        return None

    return QuarkyonicState(
        n_b=n_b,
        n_over_n0=n_b / DEFAULT_CONSTANTS.n0,
        quark_fraction=fq,
        n_q=n_q,
        n_n=n_n,
        k_bu=k_bu,
        k_f=k_f,
        energy_density=total_energy,
    )


def _minimize_state_at_density(
    model_name: str,
    a: float,
    b: float,
    n_b: float,
    settings: QuarkyonicSettings,
    par: Optional[float] = None,
) -> Optional[QuarkyonicState]:
    """Minimize the total energy density at fixed baryon density.

    The minimization is done with a coarse global scan in `f_Q` followed by
    golden-section refinement inside every candidate local-minimum bracket.
    """
    lower = _lower_quark_fraction_bound(n_b, b, settings)
    if lower >= 1.0:
        return None

    fq_grid = linspace(lower, 1.0, settings.fq_scan_points)
    trial_states: list[Optional[QuarkyonicState]] = [
        _state_at_fraction(model_name, a, b, n_b, fq, settings, par=par) for fq in fq_grid
    ]

    finite_trials = [
        (index, state) for index, state in enumerate(trial_states) if state is not None
    ]
    if not finite_trials:
        return None

    best_state = min(
        (state for _, state in finite_trials),
        key=lambda state: state.energy_density,
    )

    finite_energies = [
        state.energy_density if state is not None else float("inf")
        for state in trial_states
    ]
    candidate_intervals: list[tuple[float, float]] = []

    if len(fq_grid) >= 2:
        if finite_energies[0] <= finite_energies[1]:
            candidate_intervals.append((fq_grid[0], fq_grid[1]))
        if finite_energies[-1] <= finite_energies[-2]:
            candidate_intervals.append((fq_grid[-2], fq_grid[-1]))

    for i in range(1, len(fq_grid) - 1):
        e_left = finite_energies[i - 1]
        e_mid = finite_energies[i]
        e_right = finite_energies[i + 1]
        if not math.isfinite(e_mid):
            continue
        if e_mid <= e_left and e_mid <= e_right:
            candidate_intervals.append((fq_grid[i - 1], fq_grid[i + 1]))

    if not candidate_intervals:
        return best_state

    def objective(fq_value: float) -> float:
        state = _state_at_fraction(model_name, a, b, n_b, fq_value, settings, par=par)
        if state is None:
            return float("inf")
        return state.energy_density

    for left, right in candidate_intervals:
        fq_star, _ = golden_section_min(
            objective,
            left,
            right,
            tol=settings.refine_tol,
            max_iter=settings.refine_max_iter,
        )
        refined_state = _state_at_fraction(model_name, a, b, n_b, fq_star, settings, par=par)
        if refined_state is None:
            continue
        if refined_state.energy_density < best_state.energy_density:
            best_state = refined_state

    return best_state


def compute_quarkyonic_sound_speed_curve(
    model_name: str,
    par: Optional[float] = None,
    settings: QuarkyonicSettings = DEFAULT_SETTINGS,
) -> Optional[QuarkyonicCurve]:
    """Compute one quarkyonic `v_s^2` curve using the fitted hadronic EOS."""
    ground_state = compute_ground_state_point(model_name, par=par)
    if ground_state is None:
        return None

    density_ratios = linspace(settings.n_min_ratio, settings.n_max_ratio, settings.n_points)
    states: list[QuarkyonicState] = []
    for ratio in density_ratios:
        n_b = ratio * DEFAULT_CONSTANTS.n0
        state = _minimize_state_at_density(
            model_name,
            ground_state.a,
            ground_state.b,
            n_b,
            settings,
            par=par,
        )
        if state is not None:
            states.append(state)

    if len(states) < 5:
        return None

    n_values = [state.n_b for state in states]
    n_over_n0 = [state.n_over_n0 for state in states]
    eps_raw_values = [state.energy_density for state in states]
    fq_values = [state.quark_fraction for state in states]
    n_q_values = [state.n_q for state in states]
    n_n_values = [state.n_n for state in states]
    k_bu_values = [state.k_bu for state in states]
    k_f_values = [state.k_f for state in states]

    eps_values = smooth_local_polynomial(
        n_values,
        eps_raw_values,
        settings.smoothing_window,
        settings.smoothing_degree,
    )

    mu_b_values = first_derivative_from_grid(n_values, eps_values)
    p_raw_values = [
        n_value * mu_b_value - eps_value
        for n_value, mu_b_value, eps_value in zip(n_values, mu_b_values, eps_values)
    ]
    p_values = smooth_local_polynomial(
        n_values,
        p_raw_values,
        settings.smoothing_window,
        settings.smoothing_degree,
    )
    dP_dn_values = first_derivative_from_grid(n_values, p_values)
    d2eps_values = second_derivative_from_grid(n_values, eps_values)

    vs2_values: list[float] = []
    for mu_b_value, dP_dn_value in zip(mu_b_values, dP_dn_values):
        if (
            not math.isfinite(mu_b_value)
            or not math.isfinite(dP_dn_value)
            or abs(mu_b_value) < settings.derivative_floor
        ):
            vs2_values.append(float("nan"))
            continue
        vs2_values.append(dP_dn_value / mu_b_value)

    finite_pairs = [
        (index, value) for index, value in enumerate(vs2_values) if math.isfinite(value)
    ]
    if not finite_pairs:
        return None

    peak_index, peak_value = max(finite_pairs, key=lambda item: item[1])
    n_tr = n_values[peak_index]
    n_tr_over_n0 = n_over_n0[peak_index]

    return QuarkyonicCurve(
        model=model_name,
        par=par,
        a=ground_state.a,
        b=ground_state.b,
        K0=ground_state.K0,
        n=n_values,
        n_over_n0=n_over_n0,
        eps=eps_values,
        mu_b=mu_b_values,
        P=p_values,
        dP_dn=dP_dn_values,
        d2eps_dn2=d2eps_values,
        vs2=vs2_values,
        quark_fraction=fq_values,
        n_q=n_q_values,
        n_n=n_n_values,
        k_bu=k_bu_values,
        k_f=k_f_values,
        vs2_max=peak_value,
        n_tr=n_tr,
        n_tr_over_n0=n_tr_over_n0,
    )


def compute_selected_quarkyonic_curves(
    settings: QuarkyonicSettings = DEFAULT_SETTINGS,
) -> list[QuarkyonicCurve]:
    """Compute the selected quarkyonic curves used in the summary notebook."""
    requested = [
        ("vdw", None),
        ("rks", None),
        ("pr", None),
        ("dieterici", settings.target_alpha),
        ("clausius", settings.target_c),
    ]

    curves: list[QuarkyonicCurve] = []
    for model_name, par in requested:
        curve = compute_quarkyonic_sound_speed_curve(model_name, par=par, settings=settings)
        if curve is not None:
            curves.append(curve)
    return curves
