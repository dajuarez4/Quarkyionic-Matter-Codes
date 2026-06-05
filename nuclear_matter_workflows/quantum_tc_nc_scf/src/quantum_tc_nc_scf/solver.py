"""High-level workflow for quantum `T_c` and `n_c` using SCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from ground_state_ab_k0 import DEFAULT_ALPHA_LIST, DEFAULT_C_LIST, compute_ground_state_point

from .constants import DEFAULT_SETTINGS, QuantumCriticalSettings
from .scf import QuantumPressureEvaluator, solve_critical_point_scf


@dataclass(frozen=True)
class QuantumCriticalResult:
    """Combined ground-state and quantum critical-point result."""

    model: str
    par: Optional[float]
    a: float
    b: float
    K0: float
    Tc: float
    nc: float
    Pc: float
    dPdn: float
    d2Pdn2: float
    score: float
    iterations: int
    mu_star: float


def _default_seed(model_name: str, par: Optional[float]) -> Optional[Tuple[float, float]]:
    if model_name == "vdw":
        return (19.5, 0.072)
    if model_name == "cs":
        return (18.6, 0.070)
    if model_name == "tvm":
        return (18.3, 0.069)
    if model_name == "rks":
        return (18.0, 0.064)
    if model_name == "pr":
        return (17.1, 0.061)
    if model_name == "clausius" and par is not None:
        return (16.2, 0.050)
    if model_name == "dieterici" and par is not None:
        return (13.0, 0.058)
    return None


def compute_quantum_critical_point(
    model_name: str,
    par: Optional[float] = None,
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
    preferred_seed: Optional[Tuple[float, float]] = None,
) -> Optional[QuantumCriticalResult]:
    """Compute the quantum critical point for one model."""
    ground_state = compute_ground_state_point(model_name, par=par)
    if ground_state is None:
        return None

    evaluator = QuantumPressureEvaluator(
        model_name=model_name,
        a=ground_state.a,
        b=ground_state.b,
        par=par,
        settings=settings,
    )

    seed = preferred_seed if preferred_seed is not None else _default_seed(model_name, par)
    critical_point = solve_critical_point_scf(evaluator, settings=settings, preferred_seed=seed)
    if critical_point is None:
        return None

    return QuantumCriticalResult(
        model=model_name,
        par=par,
        a=ground_state.a,
        b=ground_state.b,
        K0=ground_state.K0,
        Tc=critical_point.Tc,
        nc=critical_point.nc,
        Pc=critical_point.Pc,
        dPdn=critical_point.dPdn,
        d2Pdn2=critical_point.d2Pdn2,
        score=critical_point.score,
        iterations=critical_point.iterations,
        mu_star=critical_point.mu_star,
    )


def compute_model_family_quantum(
    model_name: str,
    parameter_values: Iterable[float],
    settings: QuantumCriticalSettings = DEFAULT_SETTINGS,
) -> List[QuantumCriticalResult]:
    """Compute a parameter sweep using continuation in the SCF seed."""
    results: List[QuantumCriticalResult] = []
    previous_seed: Optional[Tuple[float, float]] = None

    for par in parameter_values:
        result = compute_quantum_critical_point(
            model_name,
            par=par,
            settings=settings,
            preferred_seed=previous_seed,
        )
        if result is not None:
            results.append(result)
            previous_seed = (result.Tc, result.nc)

    return results
