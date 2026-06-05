"""Quantum finite-temperature critical-point solver based on SCF iterations."""

from .constants import DEFAULT_ALPHA_LIST, DEFAULT_C_LIST, DEFAULT_SETTINGS, QuantumCriticalSettings
from .reporting import critical_result_to_record, critical_results_to_records
from .solver import (
    QuantumCriticalResult,
    compute_model_family_quantum,
    compute_quantum_critical_point,
)

__all__ = [
    "DEFAULT_ALPHA_LIST",
    "DEFAULT_C_LIST",
    "DEFAULT_SETTINGS",
    "QuantumCriticalResult",
    "QuantumCriticalSettings",
    "compute_model_family_quantum",
    "compute_quantum_critical_point",
    "critical_result_to_record",
    "critical_results_to_records",
]

