"""Settings for the quarkyonic sound-speed workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuarkyonicSettings:
    """Physical and numerical settings for the quarkyonic `v_s^2` calculation."""

    lambda_momentum_mev: float = 200.0
    nc: int = 3
    quark_degeneracy: float = 4.0

    n_min_ratio: float = 1.0e-3
    n_max_ratio: float = 5.0
    n_points: int = 200

    fq_scan_points: int = 201
    fq_min_shift: float = 1.0e-8
    refine_tol: float = 1.0e-6
    refine_max_iter: int = 120

    shell_integral_points: int = 600
    quark_integral_points: int = 600
    derivative_floor: float = 1.0e-12
    smoothing_window: int = 31
    smoothing_degree: int = 3

    target_alpha: float = 5.0 / 3.0
    target_c: float = 4.74

    @staticmethod
    def quick() -> "QuarkyonicSettings":
        """Reduced resolution for fast interactive notebook checks."""
        return QuarkyonicSettings(
            n_points=140,
            fq_scan_points=141,
            shell_integral_points=320,
            quark_integral_points=320,
            smoothing_window=21,
        )


DEFAULT_SETTINGS = QuarkyonicSettings()
