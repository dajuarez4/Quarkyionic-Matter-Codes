"""Helpers to convert quarkyonic sound-speed curves into CSV-friendly rows."""

from __future__ import annotations

import math

from .solver import QuarkyonicCurve


def curve_summary_to_record(curve: QuarkyonicCurve) -> dict[str, float | str]:
    """Return a compact summary row for one curve."""
    return {
        "model": curve.model,
        "parameter_value": math.nan if curve.par is None else curve.par,
        "a": curve.a,
        "b": curve.b,
        "K0": curve.K0,
        "vs2_max": curve.vs2_max,
        "n_tr": curve.n_tr,
        "n_tr_over_n0": curve.n_tr_over_n0,
    }


def summaries_to_records(curves: list[QuarkyonicCurve]) -> list[dict[str, float | str]]:
    """Return compact summary rows for many curves."""
    return [curve_summary_to_record(curve) for curve in curves]


def curve_to_records(curve: QuarkyonicCurve) -> list[dict[str, float | str]]:
    """Flatten one full curve into one row per density value."""
    records: list[dict[str, float | str]] = []
    for (
        n,
        n_over_n0,
        eps,
        mu_b,
        pressure,
        dP_dn,
        d2eps,
        vs2,
        fq,
        n_q,
        n_n,
        k_bu,
        k_f,
    ) in zip(
        curve.n,
        curve.n_over_n0,
        curve.eps,
        curve.mu_b,
        curve.P,
        curve.dP_dn,
        curve.d2eps_dn2,
        curve.vs2,
        curve.quark_fraction,
        curve.n_q,
        curve.n_n,
        curve.k_bu,
        curve.k_f,
    ):
        records.append(
            {
                "model": curve.model,
                "parameter_value": math.nan if curve.par is None else curve.par,
                "a": curve.a,
                "b": curve.b,
                "K0": curve.K0,
                "n": n,
                "n_over_n0": n_over_n0,
                "eps": eps,
                "mu_b": mu_b,
                "P": pressure,
                "dP_dn": dP_dn,
                "d2eps_dn2": d2eps,
                "vs2": vs2,
                "quark_fraction": fq,
                "n_q": n_q,
                "n_n": n_n,
                "k_bu": k_bu,
                "k_f": k_f,
            }
        )
    return records
