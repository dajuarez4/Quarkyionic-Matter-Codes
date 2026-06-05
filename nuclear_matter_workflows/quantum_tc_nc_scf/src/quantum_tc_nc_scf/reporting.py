"""Helpers for converting quantum critical-point results into row data."""

from __future__ import annotations

import math
from typing import Dict, List

from .solver import QuantumCriticalResult


def critical_result_to_record(
    result: QuantumCriticalResult,
    parameter_name: str = "",
) -> Dict[str, float]:
    """Convert one result dataclass into a row-style dictionary."""
    parameter_value = math.nan if result.par is None else result.par
    return {
        "model": result.model,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "a": result.a,
        "b": result.b,
        "K0": result.K0,
        "Tc": result.Tc,
        "nc": result.nc,
        "Pc": result.Pc,
        "dPdn": result.dPdn,
        "d2Pdn2": result.d2Pdn2,
        "score": result.score,
        "iterations": result.iterations,
        "mu_star": result.mu_star,
    }


def critical_results_to_records(
    results: List[QuantumCriticalResult],
    parameter_name: str = "",
) -> List[Dict[str, float]]:
    """Convert a list of result objects into row dictionaries."""
    return [critical_result_to_record(result, parameter_name=parameter_name) for result in results]

