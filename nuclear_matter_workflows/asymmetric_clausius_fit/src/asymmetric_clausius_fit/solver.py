"""Asymmetric Clausius fit at zero temperature."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEFAULT_CONSTANTS, DEFAULT_K0_TARGETS, AsymmetricClausiusConstants
from .fermi_gas import eps_id_from_nid, kf_from_nid, p_id_from_nid
from .utils import bisection_root, golden_section_min, linspace, numerical_derivative


@dataclass(frozen=True)
class SymmetricClausiusPoint:
    """One symmetric Clausius solution at fixed `c`."""

    c: float
    a: float
    b: float
    K0: float
    n_id: float
    kf: float
    p_id: float
    eps_id: float
    binding: float
    n_sat: float
    e_per_particle_minus_m_at_sat: float
    p_sat: float | None


@dataclass(frozen=True)
class AsymmetricClausiusResult:
    """Full asymmetric Clausius fit at fixed target `K0`."""

    target_k0: float
    c: float
    a_avg: float
    b_avg: float
    a_n: float
    a_pn: float
    b_n: float
    b_pn: float
    ratio_a_pn_over_a_n: float
    ratio_b_pn_over_b_n: float
    J: float
    L: float
    K0: float
    binding: float
    n_sat: float
    p_sat: float | None
    j_residual: float
    l_residual: float
    k0_residual: float


def _ev_fraction(x: float) -> float | None:
    value = 1.0 - x
    if value <= 0.0:
        return None
    return value


def _symmetric_nid_from_b(
    b: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    fraction = _ev_fraction(b * constants.n0)
    if fraction is None:
        return None
    return constants.n0 / fraction


def _symmetric_a_from_bc(
    b: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    n_id = _symmetric_nid_from_b(b, constants)
    if n_id is None:
        return None
    p_id = p_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    return p_id * (1.0 + c * constants.n0) ** 2 / (constants.n0 ** 2)


def _symmetric_binding_residual_for_b(
    b: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    fraction = _ev_fraction(b * constants.n0)
    if fraction is None:
        return None

    n_id = constants.n0 / fraction
    a_value = _symmetric_a_from_bc(b, c, constants)
    if a_value is None:
        return None

    eps_id = eps_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    binding = (
        fraction * eps_id / constants.n0
        - a_value * constants.n0 / (1.0 + c * constants.n0)
        - constants.m_nucleon
    )
    return binding - constants.binding_energy


def _find_symmetric_bracket_for_b(
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> tuple[float | None, float | None]:
    previous_b: float | None = None
    previous_residual: float | None = None

    for b in linspace(constants.b_min, constants.b_max, constants.b_scan_steps):
        residual = _symmetric_binding_residual_for_b(b, c, constants)
        if residual is None:
            previous_b = None
            previous_residual = None
            continue

        if previous_b is not None and previous_residual is not None:
            if residual == 0.0:
                return b, b
            if previous_residual * residual < 0.0:
                return previous_b, b

        previous_b = b
        previous_residual = residual

    return None, None


def _solve_symmetric_b(
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    left, right = _find_symmetric_bracket_for_b(c, constants)
    if left is None or right is None:
        return None
    return bisection_root(
        lambda bb: _symmetric_binding_residual_for_b(bb, c, constants),
        left,
        right,
    )


def symmetric_clausius_energy_density(
    n: float,
    a_value: float,
    b_value: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    if n <= 0.0:
        return None

    fraction = _ev_fraction(b_value * n)
    if fraction is None:
        return None

    n_id = n / fraction
    eps_id = eps_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    return fraction * eps_id - a_value * n * n / (1.0 + c_value * n)


def symmetric_clausius_pressure(
    n: float,
    a_value: float,
    b_value: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    if n <= 0.0:
        return None

    fraction = _ev_fraction(b_value * n)
    if fraction is None:
        return None

    n_id = n / fraction
    p_id = p_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    return p_id - a_value * n * n / (1.0 + c_value * n) ** 2


def symmetric_clausius_energy_per_particle_minus_m(
    n: float,
    a_value: float,
    b_value: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    eps = symmetric_clausius_energy_density(n, a_value, b_value, c_value, constants)
    if eps is None or n <= 0.0:
        return 1.0e99
    return eps / n - constants.m_nucleon


def _find_symmetric_saturation_density(
    a_value: float,
    b_value: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> tuple[float, float, float | None]:
    def objective(nn: float) -> float:
        return symmetric_clausius_energy_per_particle_minus_m(nn, a_value, b_value, c_value, constants)

    n_sat, value = golden_section_min(
        objective,
        constants.saturation_left,
        constants.saturation_right,
    )
    p_sat = symmetric_clausius_pressure(n_sat, a_value, b_value, c_value, constants)
    return n_sat, value, p_sat


def _symmetric_k0(
    a_value: float,
    b_value: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    def pressure(nn: float) -> float:
        value = symmetric_clausius_pressure(nn, a_value, b_value, c_value, constants)
        return math.nan if value is None else value

    return 9.0 * numerical_derivative(pressure, constants.n0, constants.k0_derivative_step)


def symmetric_clausius_point_from_c(
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> SymmetricClausiusPoint | None:
    """Return the symmetric Clausius point at fixed `c`."""
    b_value = _solve_symmetric_b(c, constants)
    if b_value is None:
        return None

    a_value = _symmetric_a_from_bc(b_value, c, constants)
    n_id = _symmetric_nid_from_b(b_value, constants)
    if a_value is None or n_id is None:
        return None

    kf_value = kf_from_nid(n_id, constants.symmetric_degeneracy, constants)
    p_id_value = p_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    eps_id_value = eps_id_from_nid(n_id, constants.symmetric_degeneracy, constants)
    binding = symmetric_clausius_energy_per_particle_minus_m(constants.n0, a_value, b_value, c, constants)
    k0_value = _symmetric_k0(a_value, b_value, c, constants)
    n_sat, e_sat, p_sat = _find_symmetric_saturation_density(a_value, b_value, c, constants)

    return SymmetricClausiusPoint(
        c=c,
        a=a_value,
        b=b_value,
        K0=k0_value,
        n_id=n_id,
        kf=kf_value,
        p_id=p_id_value,
        eps_id=eps_id_value,
        binding=binding,
        n_sat=n_sat,
        e_per_particle_minus_m_at_sat=e_sat,
        p_sat=p_sat,
    )


def find_c_for_target_k0(
    target_k0: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Determine `c` from the target symmetric incompressibility."""
    previous_c: float | None = None
    previous_residual: float | None = None

    for c_value in linspace(constants.c_min, constants.c_max, constants.c_scan_steps):
        point = symmetric_clausius_point_from_c(c_value, constants)
        if point is None:
            previous_c = None
            previous_residual = None
            continue

        residual = point.K0 - target_k0
        if previous_c is not None and previous_residual is not None:
            if residual == 0.0:
                return c_value
            if previous_residual * residual < 0.0:
                return bisection_root(
                    lambda cc: (
                        None
                        if symmetric_clausius_point_from_c(cc, constants) is None
                        else symmetric_clausius_point_from_c(cc, constants).K0 - target_k0
                    ),
                    previous_c,
                    c_value,
                    tol=1.0e-8,
                )

        previous_c = c_value
        previous_residual = residual

    return None


def split_parameters_from_averages(
    a_avg: float,
    b_avg: float,
    c_value: float,
    delta_a: float,
    delta_b: float,
) -> dict[str, float]:
    """Map average and split variables to physical interaction parameters."""
    return {
        "a_n": a_avg - delta_a,
        "a_pn": a_avg + delta_a,
        "b_n": b_avg - delta_b,
        "b_pn": b_avg + delta_b,
        "c": c_value,
    }


def asymmetric_clausius_energy_density(
    n: float,
    y: float,
    *,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Return the asymmetric Clausius energy density at `T=0`."""
    if n <= 0.0 or y < 0.0 or y > 1.0:
        return None

    n_p = n * y
    n_n = n * (1.0 - y)

    x_p = b_n * n_p + b_pn * n_n
    x_n = b_pn * n_p + b_n * n_n
    f_p = _ev_fraction(x_p)
    f_n = _ev_fraction(x_n)
    if f_p is None or f_n is None:
        return None

    eps_p = f_p * eps_id_from_nid(n_p / f_p, constants.species_degeneracy, constants)
    eps_n = f_n * eps_id_from_nid(n_n / f_n, constants.species_degeneracy, constants)

    den = 1.0 + c * n
    if den <= 0.0:
        return None

    attractive_num = a_n * (n_p * n_p + n_n * n_n) + 2.0 * a_pn * n_p * n_n
    return eps_p + eps_n - attractive_num / den


def asymmetric_clausius_energy_per_particle_minus_m(
    n: float,
    y: float,
    *,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Return `E/A - m` for asymmetric matter."""
    eps = asymmetric_clausius_energy_density(
        n,
        y,
        a_n=a_n,
        a_pn=a_pn,
        b_n=b_n,
        b_pn=b_pn,
        c=c,
        constants=constants,
    )
    if eps is None or n <= 0.0:
        return None
    return eps / n - constants.m_nucleon


def symmetry_energy_parabolic(
    n: float,
    *,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Return `S(n) = E_PNM(n) - E_SNM(n)`."""
    e_pnm = asymmetric_clausius_energy_per_particle_minus_m(
        n,
        0.0,
        a_n=a_n,
        a_pn=a_pn,
        b_n=b_n,
        b_pn=b_pn,
        c=c,
        constants=constants,
    )
    e_snm = asymmetric_clausius_energy_per_particle_minus_m(
        n,
        0.5,
        a_n=a_n,
        a_pn=a_pn,
        b_n=b_n,
        b_pn=b_pn,
        c=c,
        constants=constants,
    )
    if e_pnm is None or e_snm is None:
        return None
    return e_pnm - e_snm


def symmetry_slope_l(
    *,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    c: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float:
    """Return `L = 3 n0 dS/dn |_(n0)`."""

    def symmetry_at_density(nn: float) -> float:
        value = symmetry_energy_parabolic(
            nn,
            a_n=a_n,
            a_pn=a_pn,
            b_n=b_n,
            b_pn=b_pn,
            c=c,
            constants=constants,
        )
        return math.nan if value is None else value

    return 3.0 * constants.n0 * numerical_derivative(
        symmetry_at_density,
        constants.n0,
        constants.l_derivative_step,
    )


def _solve_delta_a_for_j(
    a_avg: float,
    b_avg: float,
    c_value: float,
    delta_b: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> float | None:
    limit = 0.995 * a_avg
    previous_da: float | None = None
    previous_residual: float | None = None

    for delta_a in linspace(-limit, limit, 401):
        params = split_parameters_from_averages(a_avg, b_avg, c_value, delta_a, delta_b)
        if min(params["a_n"], params["a_pn"], params["b_n"], params["b_pn"]) <= 0.0:
            previous_da = None
            previous_residual = None
            continue

        j_value = symmetry_energy_parabolic(constants.n0, constants=constants, **params)
        if j_value is None:
            previous_da = None
            previous_residual = None
            continue

        residual = j_value - constants.symmetry_energy_j
        if previous_da is not None and previous_residual is not None:
            if residual == 0.0:
                return delta_a
            if previous_residual * residual < 0.0:
                return bisection_root(
                    lambda daa: (
                        None
                        if min(
                            split_parameters_from_averages(a_avg, b_avg, c_value, daa, delta_b)["a_n"],
                            split_parameters_from_averages(a_avg, b_avg, c_value, daa, delta_b)["a_pn"],
                            split_parameters_from_averages(a_avg, b_avg, c_value, daa, delta_b)["b_n"],
                            split_parameters_from_averages(a_avg, b_avg, c_value, daa, delta_b)["b_pn"],
                        )
                        <= 0.0
                        else symmetry_energy_parabolic(
                            constants.n0,
                            constants=constants,
                            **split_parameters_from_averages(a_avg, b_avg, c_value, daa, delta_b),
                        )
                        - constants.symmetry_energy_j
                    ),
                    previous_da,
                    delta_a,
                    tol=1.0e-8,
                )

        previous_da = delta_a
        previous_residual = residual

    return None


def _solve_delta_b_for_l(
    a_avg: float,
    b_avg: float,
    c_value: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> tuple[float | None, float | None]:
    limit = 0.995 * b_avg
    previous_db: float | None = None
    previous_residual: float | None = None

    for delta_b in linspace(-limit, limit, 401):
        if b_avg - delta_b <= 0.0 or b_avg + delta_b <= 0.0:
            previous_db = None
            previous_residual = None
            continue

        delta_a = _solve_delta_a_for_j(a_avg, b_avg, c_value, delta_b, constants)
        if delta_a is None:
            previous_db = None
            previous_residual = None
            continue

        params = split_parameters_from_averages(a_avg, b_avg, c_value, delta_a, delta_b)
        l_value = symmetry_slope_l(constants=constants, **params)
        if math.isnan(l_value):
            previous_db = None
            previous_residual = None
            continue

        residual = l_value - constants.symmetry_slope_l
        if previous_db is not None and previous_residual is not None:
            if residual == 0.0:
                return delta_a, delta_b
            if previous_residual * residual < 0.0:
                root_db = bisection_root(
                    lambda dbb: (
                        None
                        if b_avg - dbb <= 0.0 or b_avg + dbb <= 0.0
                        else (
                            None
                            if _solve_delta_a_for_j(a_avg, b_avg, c_value, dbb, constants) is None
                            else symmetry_slope_l(
                                constants=constants,
                                **split_parameters_from_averages(
                                    a_avg,
                                    b_avg,
                                    c_value,
                                    _solve_delta_a_for_j(a_avg, b_avg, c_value, dbb, constants),
                                    dbb,
                                ),
                            )
                            - constants.symmetry_slope_l
                        )
                    ),
                    previous_db,
                    delta_b,
                    tol=1.0e-8,
                )
                if root_db is None:
                    return None, None
                root_da = _solve_delta_a_for_j(a_avg, b_avg, c_value, root_db, constants)
                return root_da, root_db

        previous_db = delta_b
        previous_residual = residual

    return None, None


def fit_asymmetric_clausius_target(
    target_k0: float,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> AsymmetricClausiusResult | None:
    """Solve the asymmetric Clausius fit for one target `K0`."""
    c_value = find_c_for_target_k0(target_k0, constants)
    if c_value is None:
        return None

    symmetric_point = symmetric_clausius_point_from_c(c_value, constants)
    if symmetric_point is None:
        return None

    delta_a, delta_b = _solve_delta_b_for_l(symmetric_point.a, symmetric_point.b, c_value, constants)
    if delta_a is None or delta_b is None:
        return None

    params = split_parameters_from_averages(symmetric_point.a, symmetric_point.b, c_value, delta_a, delta_b)
    j_value = symmetry_energy_parabolic(constants.n0, constants=constants, **params)
    if j_value is None:
        return None
    l_value = symmetry_slope_l(constants=constants, **params)

    return AsymmetricClausiusResult(
        target_k0=target_k0,
        c=c_value,
        a_avg=symmetric_point.a,
        b_avg=symmetric_point.b,
        a_n=params["a_n"],
        a_pn=params["a_pn"],
        b_n=params["b_n"],
        b_pn=params["b_pn"],
        ratio_a_pn_over_a_n=params["a_pn"] / params["a_n"],
        ratio_b_pn_over_b_n=params["b_pn"] / params["b_n"],
        J=j_value,
        L=l_value,
        K0=symmetric_point.K0,
        binding=symmetric_point.binding,
        n_sat=symmetric_point.n_sat,
        p_sat=symmetric_point.p_sat,
        j_residual=j_value - constants.symmetry_energy_j,
        l_residual=l_value - constants.symmetry_slope_l,
        k0_residual=symmetric_point.K0 - target_k0,
    )


def compute_target_family(
    target_k0_values: tuple[float, ...] | list[float] = DEFAULT_K0_TARGETS,
    constants: AsymmetricClausiusConstants = DEFAULT_CONSTANTS,
) -> list[AsymmetricClausiusResult]:
    """Compute the asymmetric Clausius fit for a full `K0` target list."""
    results: list[AsymmetricClausiusResult] = []
    for target_k0 in target_k0_values:
        result = fit_asymmetric_clausius_target(float(target_k0), constants)
        if result is not None:
            results.append(result)
    return results
