"""Helpers to convert solver results into row-oriented table data."""

from __future__ import annotations

import math

from .solver import GroundStateResult


def result_to_record(result: GroundStateResult, parameter_name: str = "") -> dict[str, float | str]:
    """Convert one result dataclass into a dictionary for tabular output."""
    parameter_value = math.nan if result.par is None else result.par

    return {
        "model": result.model,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "a": result.a,
        "b": result.b,
        "K0": result.K0,
        "n_id": result.n_id,
        "kf": result.kf,
        "p_id": result.p_id,
        "eps_id": result.eps_id,
        "binding": result.binding,
        "n_sat": result.n_sat,
        "e_per_particle_minus_m_at_sat": result.e_per_particle_minus_m_at_sat,
        "p_sat": math.nan if result.p_sat is None else result.p_sat,
    }


def results_to_records(
    results: list[GroundStateResult],
    parameter_name: str = "",
) -> list[dict[str, float | str]]:
    """Convert a list of result objects into a list of dictionaries."""
    return [result_to_record(result, parameter_name=parameter_name) for result in results]

