"""Constants and default parameter grids for the ground-state solver."""

from __future__ import annotations

from dataclasses import dataclass


def _make_grid(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (float(start),)

    step = (stop - start) / float(count - 1)
    return tuple(start + step * float(i) for i in range(count))


@dataclass(frozen=True)
class GroundStateConstants:
    """Physical and numerical constants used by the `T=0` workflow."""

    hbarc: float = 197.3269804
    m_nuc: float = 938.0
    dgen: float = 4.0
    n0: float = 0.16
    w0: float = -16.0

    b_min: float = 1.0e-6
    b_max: float = 6.0
    b_scan_steps: int = 5000

    n_int: int = 4000
    k0_derivative_step: float = 1.0e-6

    saturation_left: float = 0.05
    saturation_right: float = 0.30


DEFAULT_CONSTANTS = GroundStateConstants()
DEFAULT_ALPHA_LIST = _make_grid(5.0 / 3.0, 2.0, 10)
DEFAULT_C_LIST = _make_grid(0.0, 4.74, 10)

