"""Ground-state solver for the interaction parameters `a`, `b`, and `K0`."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEFAULT_CONSTANTS, GroundStateConstants
from .fermi_gas import eps_id_from_nid, kf_from_nid, p_id_from_nid
from .models import MODELS
from .utils.numerics import bisection_root, golden_section_min, linspace, numerical_derivative


@dataclass(frozen=True)
class GroundStateResult:
    """Final ground-state output for one model and one parameter value."""

    model: str
    par: float | None
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


def n_id_ground_from_b(
    model_name: str,
    b: float,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
) -> float | None:
    """Map the physical density `n0` to the ideal-gas density for the model."""
    model = MODELS[model_name]
    density = constants.n0 if n0 is None else n0
    return model.nid_from_n(density, b)


def pressure_zero_a(
    model_name: str,
    b: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
) -> float | None:
    """Compute `a` from the zero-pressure ground-state condition."""
    model = MODELS[model_name]
    density = constants.n0 if n0 is None else n0

    n_id = n_id_ground_from_b(model_name, b, constants=constants, n0=density)
    if n_id is None:
        return None

    p_id = p_id_from_nid(n_id, constants)
    pressure_prefactor = model.pressure_prefactor(density, b)
    if pressure_prefactor is None:
        return None

    dU0 = model.dU(density, b, par)
    if dU0 is None:
        return None

    denom = density ** 2 * dU0
    if abs(denom) < 1.0e-14:
        return None

    return -(pressure_prefactor * p_id) / denom


def binding_energy_per_particle(
    model_name: str,
    a: float,
    b: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
) -> float | None:
    """Return `E/A - m` at the reference density."""
    model = MODELS[model_name]
    density = constants.n0 if n0 is None else n0

    n_id = n_id_ground_from_b(model_name, b, constants=constants, n0=density)
    if n_id is None:
        return None

    eps_id = eps_id_from_nid(n_id, constants)
    U0 = model.U(density, b, par)
    if U0 is None:
        return None

    return eps_id / n_id + a * U0 - constants.m_nuc


def binding_residual_for_b(
    model_name: str,
    b: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
    w0: float | None = None,
) -> float | None:
    """Residual used to determine `b` from the binding-energy condition."""
    density = constants.n0 if n0 is None else n0
    binding_target = constants.w0 if w0 is None else w0

    a = pressure_zero_a(model_name, b, par=par, constants=constants, n0=density)
    if a is None:
        return None

    w = binding_energy_per_particle(model_name, a, b, par=par, constants=constants, n0=density)
    if w is None:
        return None

    return w - binding_target


def find_bracket_for_b(
    model_name: str,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    bmin: float | None = None,
    bmax: float | None = None,
    steps: int | None = None,
) -> tuple[float | None, float | None]:
    """Scan a grid in `b` and return a sign-change bracket for the residual."""
    lower = constants.b_min if bmin is None else bmin
    upper = constants.b_max if bmax is None else bmax
    num_steps = constants.b_scan_steps if steps is None else steps

    grid = linspace(lower, upper, num_steps)
    prev_b: float | None = None
    prev_f: float | None = None

    for b in grid:
        residual = binding_residual_for_b(model_name, b, par=par, constants=constants)

        if residual is None:
            prev_b = None
            prev_f = None
            continue

        if prev_b is not None and prev_f is not None:
            if residual == 0.0:
                return b, b
            if prev_f * residual < 0.0:
                return prev_b, b

        prev_b = b
        prev_f = residual

    return None, None


def solve_parameters(
    model_name: str,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
    w0: float | None = None,
) -> dict[str, float | str | None] | None:
    """Solve the ground-state fit and return `a` and `b`."""
    density = constants.n0 if n0 is None else n0
    binding_target = constants.w0 if w0 is None else w0

    b_left, b_right = find_bracket_for_b(model_name, par=par, constants=constants)
    if b_left is None or b_right is None:
        return None

    def residual(bb: float) -> float | None:
        return binding_residual_for_b(
            model_name,
            bb,
            par=par,
            constants=constants,
            n0=density,
            w0=binding_target,
        )

    b_sol = bisection_root(residual, b_left, b_right)
    if b_sol is None:
        return None

    a_sol = pressure_zero_a(model_name, b_sol, par=par, constants=constants, n0=density)
    if a_sol is None:
        return None

    n_id_sol = n_id_ground_from_b(model_name, b_sol, constants=constants, n0=density)
    if n_id_sol is None:
        return None

    kf_sol = kf_from_nid(n_id_sol, constants)
    p_id_sol = p_id_from_nid(n_id_sol, constants)
    eps_id_sol = eps_id_from_nid(n_id_sol, constants)
    w_sol = binding_energy_per_particle(model_name, a_sol, b_sol, par=par, constants=constants, n0=density)
    if w_sol is None:
        return None

    return {
        "model": model_name,
        "par": par,
        "a": a_sol,
        "b": b_sol,
        "n_id": n_id_sol,
        "kf": kf_sol,
        "p_id": p_id_sol,
        "eps_id": eps_id_sol,
        "binding": w_sol,
    }


def eps_total(
    model_name: str,
    a: float,
    b: float,
    n: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Total `T=0` energy density at baryon density `n`."""
    model = MODELS[model_name]
    if n <= 0.0:
        return None

    n_id = model.nid_from_n(n, b)
    if n_id is None:
        return None

    volume_fraction = model.volume_fraction(n, b)
    if volume_fraction is None:
        return None

    eps_id = eps_id_from_nid(n_id, constants)
    U_value = model.U(n, b, par)
    if U_value is None:
        return None

    return volume_fraction * eps_id + n * a * U_value


def pressure_total(
    model_name: str,
    a: float,
    b: float,
    n: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
) -> float | None:
    """Total `T=0` pressure at baryon density `n`."""
    model = MODELS[model_name]
    if n <= 0.0:
        return None

    n_id = model.nid_from_n(n, b)
    if n_id is None:
        return None

    pressure_prefactor = model.pressure_prefactor(n, b)
    if pressure_prefactor is None:
        return None

    p_id = p_id_from_nid(n_id, constants)
    dU_value = model.dU(n, b, par)
    if dU_value is None:
        return None

    return pressure_prefactor * p_id + n * n * a * dU_value


def energy_per_particle_minus_m(
    model_name: str,
    a: float,
    b: float,
    n: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
) -> float:
    """Objective function used for the saturation-density check."""
    eps = eps_total(model_name, a, b, n, par=par, constants=constants)
    if eps is None or n <= 0.0:
        return 1.0e99
    return eps / n - constants.m_nuc


def find_saturation_density(
    model_name: str,
    a: float,
    b: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n_left: float | None = None,
    n_right: float | None = None,
) -> tuple[float, float, float | None]:
    """Locate the minimum of `E/A - m` as a consistency check."""
    left = constants.saturation_left if n_left is None else n_left
    right = constants.saturation_right if n_right is None else n_right

    def objective(n: float) -> float:
        return energy_per_particle_minus_m(model_name, a, b, n, par=par, constants=constants)

    n_sat, value = golden_section_min(objective, left, right)
    p_sat = pressure_total(model_name, a, b, n_sat, par=par, constants=constants)
    return n_sat, value, p_sat


def incompressibility_k0(
    model_name: str,
    a: float,
    b: float,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
    n0: float | None = None,
) -> float:
    """Compute the incompressibility `K0 = 9 dP/dn` at `n0`."""
    density = constants.n0 if n0 is None else n0

    def pressure_at_density(nn: float) -> float:
        value = pressure_total(model_name, a, b, nn, par=par, constants=constants)
        return math.nan if value is None else value

    return 9.0 * numerical_derivative(pressure_at_density, density, h=constants.k0_derivative_step)


def compute_ground_state_point(
    model_name: str,
    par: float | None = None,
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
) -> GroundStateResult | None:
    """Compute `a`, `b`, and `K0` for one model."""
    solution = solve_parameters(model_name, par=par, constants=constants)
    if solution is None:
        return None

    a_value = float(solution["a"])
    b_value = float(solution["b"])
    k0_value = incompressibility_k0(model_name, a_value, b_value, par=par, constants=constants)
    n_sat, e_sat, p_sat = find_saturation_density(
        model_name,
        a_value,
        b_value,
        par=par,
        constants=constants,
    )

    return GroundStateResult(
        model=str(solution["model"]),
        par=par,
        a=a_value,
        b=b_value,
        K0=k0_value,
        n_id=float(solution["n_id"]),
        kf=float(solution["kf"]),
        p_id=float(solution["p_id"]),
        eps_id=float(solution["eps_id"]),
        binding=float(solution["binding"]),
        n_sat=n_sat,
        e_per_particle_minus_m_at_sat=e_sat,
        p_sat=p_sat,
    )


def compute_model_family(
    model_name: str,
    parameter_values: list[float] | tuple[float, ...],
    constants: GroundStateConstants = DEFAULT_CONSTANTS,
) -> list[GroundStateResult]:
    """Compute a full parameter sweep for one model family."""
    results: list[GroundStateResult] = []
    for par in parameter_values:
        point = compute_ground_state_point(model_name, par=par, constants=constants)
        if point is not None:
            results.append(point)
    return results
