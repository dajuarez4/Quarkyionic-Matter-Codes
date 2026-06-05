"""Quarkyonic EOS workflow driven by the asymmetric Clausius fit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


G = 2.0
NC = 3.0

MP = 0.938
MN = 0.938
ME = 0.000511
MMU = 0.10566

MU = MP / NC
MD = MP / NC
LAM = 0.2

RHO0 = 1.23e-3
GEV4_TO_MEV_FM3 = 1.30148926289e5
FM3_TO_GEV_MINUS3 = (1.0 / 0.1973269804) ** 3
MEV_FM3_TO_GEV_MINUS2 = 1.0e-3 * FM3_TO_GEV_MINUS3


def default_rhob_list(n_points: int = 80, low: float = 1.0e-6, high: float = 5.0) -> np.ndarray:
    """Return the baryon-density grid in GeV^3 units."""
    return np.linspace(low, high, n_points) * RHO0


def default_y_values() -> list[float]:
    """Return the default proton-fraction values used in fixed-y scans."""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def integrate_simpson(func, a: float, b: float, n: int = 1000) -> float:
    """Compute a 1D integral with Simpson's rule."""
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if b < a:
        return 0.0
    if abs(b - a) < 1.0e-14:
        return 0.0
    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    total = func(a) + func(b)
    x = a
    for i in range(1, n):
        x += h
        total += (4.0 if i % 2 == 1 else 2.0) * func(x)
    return total * h / 3.0


def gss_safe(func, a: float, b: float, tol: float = 1.0e-6, max_iter: int = 400) -> tuple[float, float]:
    """Golden-section minimization that tolerates `nan` values."""
    gr = 0.5 * (math.sqrt(5.0) + 1.0)
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = func(c)
    fd = func(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if (not np.isfinite(fc)) and (not np.isfinite(fd)):
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            fc = func(c)
            fd = func(d)
            continue

        if not np.isfinite(fc):
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = func(d)
            continue

        if not np.isfinite(fd):
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = func(c)
            continue

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = func(d)

    x_star = 0.5 * (a + b)
    return float(x_star), float(func(x_star))


def convert_fit_df_to_gev_units(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Convert fitted Clausius parameters from fm/MeV units to GeV units."""
    converted = fit_df.copy()
    for column in ["a_avg", "a_n", "a_pn"]:
        if column in converted.columns:
            converted[f"{column}_gev"] = converted[column] * MEV_FM3_TO_GEV_MINUS2
    for column in ["b_avg", "b_n", "b_pn", "c"]:
        if column in converted.columns:
            converted[f"{column}_gev"] = converted[column] * FM3_TO_GEV_MINUS3
    return converted


def fit_df_to_parameter_dict(
    fit_df: pd.DataFrame,
    target_k0_values: Iterable[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Build a parameter dictionary keyed by `k0_<value>` labels."""
    converted = convert_fit_df_to_gev_units(fit_df)
    if target_k0_values is not None:
        target_set = {float(value) for value in target_k0_values}
        converted = converted[converted["target_k0"].astype(float).isin(target_set)].copy()

    parameters: dict[str, dict[str, float]] = {}
    for row in converted.itertuples(index=False):
        label = f"k0_{int(round(float(row.target_k0)))}"
        parameters[label] = {
            "target_k0": float(row.target_k0),
            "a_n": float(row.a_n_gev),
            "a_pn": float(row.a_pn_gev),
            "b_n": float(row.b_n_gev),
            "b_pn": float(row.b_pn_gev),
            "c": float(row.c_gev),
            "a_n_fm": float(row.a_n),
            "a_pn_fm": float(row.a_pn),
            "b_n_fm": float(row.b_n),
            "b_pn_fm": float(row.b_pn),
            "c_fm": float(row.c),
        }
    return parameters


class AsymmetricClausiusQuarkyonicEOS:
    """Quarkyonic EOS with asymmetric Clausius hadronic interactions."""

    def __init__(
        self,
        label: str,
        params: dict[str, float],
        *,
        g: float = G,
        nc: float = NC,
        mp: float = MP,
        mn: float = MN,
        me: float = ME,
        mmu: float = MMU,
        mu: float = MU,
        md: float = MD,
        lam: float = LAM,
        rho0: float = RHO0,
        nint: int = 300,
    ):
        self.label = label
        self.params = dict(params)
        self.g = g
        self.nc = nc
        self.mp = mp
        self.mn = mn
        self.me = me
        self.mmu = mmu
        self.mu = mu
        self.md = md
        self.lam = lam
        self.rho0 = rho0
        self.nint = nint

    @property
    def target_k0(self) -> float:
        return float(self.params["target_k0"])

    @property
    def a_n(self) -> float:
        return float(self.params["a_n"])

    @property
    def a_pn(self) -> float:
        return float(self.params["a_pn"])

    @property
    def b_n(self) -> float:
        return float(self.params["b_n"])

    @property
    def b_pn(self) -> float:
        return float(self.params["b_pn"])

    @property
    def c(self) -> float:
        return float(self.params["c"])

    def _hadronic_densities(self, rhob: float, fq: float, y: float) -> tuple[float, float]:
        n_h = rhob * (1.0 - fq)
        return n_h * y, n_h * (1.0 - y)

    def _volume_fractions(self, rhob: float, fq: float, y: float) -> tuple[float, float]:
        n_p, n_n = self._hadronic_densities(rhob, fq, y)
        x_p = self.b_n * n_p + self.b_pn * n_n
        x_n = self.b_pn * n_p + self.b_n * n_n
        return 1.0 - x_p, 1.0 - x_n

    def kbu_solver(self, rhob: float, fq: float, y: float) -> float:
        pref = self.g / (2.0 * np.pi ** 2)
        w = (3.0 / (2.0 - y)) * pref
        nq = rhob * fq
        inside = (3.0 * nq / w + self.lam ** 3) ** (2.0 / 3.0) - self.lam ** 2
        return self.nc * np.sqrt(max(inside, 0.0))

    def euid(self, kbu: float, y: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(q: float) -> float:
            return q * np.sqrt(self.lam ** 2 + q ** 2) * np.sqrt(self.mu ** 2 + q ** 2)

        result = (1.0 + y) / (2.0 - y) * integrate_simpson(integrand, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def edid(self, kbu: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(q: float) -> float:
            return q * np.sqrt(self.lam ** 2 + q ** 2) * np.sqrt(self.md ** 2 + q ** 2)

        result = integrate_simpson(integrand, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def enid(self, kbu: float, kfn: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(k: float) -> float:
            return k ** 2 * np.sqrt(self.mn ** 2 + k ** 2)

        result = integrate_simpson(integrand, kbu, kfn, n=nint)
        return pref * result

    def epid(self, kbu: float, kfp: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(k: float) -> float:
            return k ** 2 * np.sqrt(self.mp ** 2 + k ** 2)

        result = integrate_simpson(integrand, kbu, kfp, n=nint)
        return pref * result

    def nuid(self, kbu: float, y: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(q: float) -> float:
            return q * np.sqrt(self.lam ** 2 + q ** 2)

        result = (1.0 + y) / (2.0 - y) * integrate_simpson(integrand, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def ndid(self, kbu: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi ** 2)

        def integrand(q: float) -> float:
            return q * np.sqrt(self.lam ** 2 + q ** 2)

        result = integrate_simpson(integrand, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def kfp_solver(self, rhob: float, fq: float, y: float, kbu: float) -> float:
        n_p, _ = self._hadronic_densities(rhob, fq, y)
        if n_p <= 0.0:
            return kbu
        f_p, _ = self._volume_fractions(rhob, fq, y)
        if f_p <= 0.0 or not np.isfinite(f_p):
            return np.nan
        n_p_id = n_p / f_p
        val = kbu ** 3 + (6.0 * np.pi ** 2 / self.g) * n_p_id
        return np.cbrt(max(val, 0.0))

    def kfn_solver(self, rhob: float, fq: float, y: float, kbu: float) -> float:
        _, n_n = self._hadronic_densities(rhob, fq, y)
        if n_n <= 0.0:
            return kbu
        _, f_n = self._volume_fractions(rhob, fq, y)
        if f_n <= 0.0 or not np.isfinite(f_n):
            return np.nan
        n_n_id = n_n / f_n
        val = kbu ** 3 + (6.0 * np.pi ** 2 / self.g) * n_n_id
        return np.cbrt(max(val, 0.0))

    def energy_hq_of_fq_y(self, fq: float, rhob: float, y: float) -> float:
        if y < 0.0 or y > 0.5:
            return np.nan

        kbu = self.kbu_solver(rhob, fq, y)
        kfp = self.kfp_solver(rhob, fq, y, kbu)
        kfn = self.kfn_solver(rhob, fq, y, kbu)
        if (not np.isfinite(kbu)) or (not np.isfinite(kfp)) or (not np.isfinite(kfn)):
            return np.nan

        eu = self.euid(kbu, y=y)
        ed = self.edid(kbu)
        en = self.enid(kbu, kfn)
        ep = self.epid(kbu, kfp)

        n_p, n_n = self._hadronic_densities(rhob, fq, y)
        n_h = n_p + n_n
        f_p, f_n = self._volume_fractions(rhob, fq, y)
        if n_p > 0.0 and ((not np.isfinite(f_p)) or f_p <= 0.0):
            return np.nan
        if n_n > 0.0 and ((not np.isfinite(f_n)) or f_n <= 0.0):
            return np.nan

        den = 1.0 + self.c * n_h
        if den <= 0.0:
            return np.nan

        attractive_num = self.a_n * (n_p ** 2 + n_n ** 2) + 2.0 * self.a_pn * n_p * n_n
        return eu + ed + f_p * ep + f_n * en - attractive_num / den

    def lepton_kf_from_mu(self, mu_l: float, m_l: float) -> float:
        return np.sqrt(max(mu_l ** 2 - m_l ** 2, 0.0))

    def lepton_density_from_mu(self, mu_l: float, m_l: float) -> float:
        if mu_l <= m_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, m_l)
        return kf ** 3 / (3.0 * np.pi ** 2)

    def lepton_energy_density_from_mu(self, mu_l: float, m_l: float, nint: int | None = None) -> float:
        if nint is None:
            nint = self.nint
        if mu_l <= m_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, m_l)

        def integrand(k: float) -> float:
            return k ** 2 * np.sqrt(k ** 2 + m_l ** 2)

        return (1.0 / np.pi ** 2) * integrate_simpson(integrand, 0.0, kf, n=nint)

    def charge_density_hq(self, rhob: float, fq: float, y: float) -> float:
        kbu = self.kbu_solver(rhob, fq, y)
        if not np.isfinite(kbu):
            return np.nan

        n_p, _ = self._hadronic_densities(rhob, fq, y)
        n_u = 3.0 * self.nuid(kbu, y)
        n_d = 3.0 * self.ndid(kbu)
        return n_p + (2.0 / 3.0) * n_u - (1.0 / 3.0) * n_d

    def mu_q_fd(self, rhob: float, fq: float, y: float, dy: float = 1.0e-5) -> float:
        y1 = max(1.0e-8, y - dy)
        y2 = min(0.5 - 1.0e-8, y + dy)
        e1 = self.energy_hq_of_fq_y(fq, rhob, y1)
        e2 = self.energy_hq_of_fq_y(fq, rhob, y2)
        if (not np.isfinite(e1)) or (not np.isfinite(e2)):
            return np.nan
        return (e2 - e1) / (rhob * (y2 - y1))

    def mu_e_from_y(self, rhob: float, fq: float, y: float) -> float:
        mu_q = self.mu_q_fd(rhob, fq, y)
        if not np.isfinite(mu_q):
            return np.nan
        return -mu_q

    def neutrality_residual(self, y: float, rhob: float, fq: float) -> float:
        mu_e = self.mu_e_from_y(rhob, fq, y)
        if not np.isfinite(mu_e):
            return np.nan

        rho_hq = self.charge_density_hq(rhob, fq, y)
        rho_e = self.lepton_density_from_mu(mu_e, self.me)
        rho_mu = self.lepton_density_from_mu(mu_e, self.mmu)
        if not np.isfinite(rho_hq):
            return np.nan
        return rho_hq - (rho_e + rho_mu)

    def solve_y_beta_equilibrium(
        self,
        rhob: float,
        fq: float,
        y_min: float = 1.0e-6,
        y_max: float = 0.5 - 1.0e-6,
        nscan: int = 160,
    ) -> tuple[float, float, float, float]:
        ys = np.linspace(y_min, y_max, nscan)
        vals = np.asarray([self.neutrality_residual(y, rhob, fq) for y in ys], dtype=float)

        y_star = np.nan
        for i in range(len(ys) - 1):
            left = ys[i]
            right = ys[i + 1]
            f_left = vals[i]
            f_right = vals[i + 1]

            if (not np.isfinite(f_left)) or (not np.isfinite(f_right)):
                continue
            if abs(f_left) < 1.0e-12:
                y_star = left
                break
            if f_left * f_right >= 0.0:
                continue

            for _ in range(200):
                mid = 0.5 * (left + right)
                f_mid = self.neutrality_residual(mid, rhob, fq)
                if not np.isfinite(f_mid):
                    break
                if abs(f_mid) < 1.0e-12:
                    left = mid
                    right = mid
                    break
                if f_left * f_mid <= 0.0:
                    right = mid
                    f_right = f_mid
                else:
                    left = mid
                    f_left = f_mid

            y_star = 0.5 * (left + right)
            break

        if not np.isfinite(y_star):
            return np.nan, np.nan, np.nan, np.nan

        mu_e = self.mu_e_from_y(rhob, fq, y_star)
        eps_e = self.lepton_energy_density_from_mu(mu_e, self.me)
        eps_mu = self.lepton_energy_density_from_mu(mu_e, self.mmu)
        return y_star, mu_e, eps_e, eps_mu

    def total_energy_beta(self, fq: float, rhob: float, y_scan: int = 160) -> float:
        y_star, _, eps_e, eps_mu = self.solve_y_beta_equilibrium(rhob, fq, nscan=y_scan)
        if not np.isfinite(y_star):
            return np.nan
        eps_hq = self.energy_hq_of_fq_y(fq, rhob, y_star)
        if not np.isfinite(eps_hq):
            return np.nan
        return eps_hq + eps_e + eps_mu

    def solve_fq_fixed_y(
        self,
        rhob: float,
        y: float,
        fq_min: float = 1.0e-3,
        fq_max: float = 1.0,
        tol: float = 1.0e-6,
    ) -> tuple[float, float]:
        return gss_safe(lambda fq: self.energy_hq_of_fq_y(fq, rhob, y), fq_min, fq_max, tol=tol)

    def solve_fq_beta(
        self,
        rhob: float,
        fq_min: float = 1.0e-3,
        fq_max: float = 1.0,
        tol: float = 1.0e-6,
        y_scan: int = 160,
    ) -> tuple[float, float]:
        return gss_safe(lambda fq: self.total_energy_beta(fq, rhob, y_scan=y_scan), fq_min, fq_max, tol=tol)

    def beta_equilibrium_profile(
        self,
        rhob_list: np.ndarray,
        fq_min: float = 1.0e-3,
        fq_max: float = 1.0,
        tol: float = 1.0e-6,
        y_scan: int = 160,
    ) -> pd.DataFrame:
        rows = []
        for rhob in rhob_list:
            fq_star, e_star = self.solve_fq_beta(rhob, fq_min=fq_min, fq_max=fq_max, tol=tol, y_scan=y_scan)
            y_star, mu_e, eps_e, eps_mu = self.solve_y_beta_equilibrium(rhob, fq_star, nscan=y_scan)
            rows.append(
                {
                    "rhob": rhob,
                    "rho_over_rho0": rhob / self.rho0,
                    "fq_star": fq_star,
                    "y_star": y_star,
                    "mu_e": mu_e,
                    "eps_e": eps_e,
                    "eps_mu": eps_mu,
                    "energy_density": e_star,
                    "target_k0": self.target_k0,
                }
            )

        df = pd.DataFrame(rows)
        energy = df["energy_density"].to_numpy(dtype=float)
        density = df["rhob"].to_numpy(dtype=float)
        mu_b = np.gradient(energy, density, edge_order=2)
        mu_bb = np.gradient(mu_b, density, edge_order=2)
        pressure = density * mu_b - energy

        df["mu_b"] = mu_b
        df["pressure"] = pressure
        df["vs2"] = density / mu_b * mu_bb
        df["energy_density_mevfm3"] = df["energy_density"] * GEV4_TO_MEV_FM3
        df["pressure_mevfm3"] = df["pressure"] * GEV4_TO_MEV_FM3
        return df

    def fixed_y_profile(
        self,
        rhob_list: np.ndarray,
        y: float,
        fq_min: float = 1.0e-3,
        fq_max: float = 1.0,
        tol: float = 1.0e-6,
    ) -> pd.DataFrame:
        rows = []
        for rhob in rhob_list:
            fq_star, e_star = self.solve_fq_fixed_y(rhob, y, fq_min=fq_min, fq_max=fq_max, tol=tol)
            rows.append(
                {
                    "rhob": rhob,
                    "rho_over_rho0": rhob / self.rho0,
                    "y": y,
                    "fq_star": fq_star,
                    "energy_density": e_star,
                    "target_k0": self.target_k0,
                }
            )

        df = pd.DataFrame(rows)
        energy = df["energy_density"].to_numpy(dtype=float)
        density = df["rhob"].to_numpy(dtype=float)
        mu_b = np.gradient(energy, density, edge_order=2)
        mu_bb = np.gradient(mu_b, density, edge_order=2)
        pressure = density * mu_b - energy

        df["mu_b"] = mu_b
        df["pressure"] = pressure
        df["vs2"] = density / mu_b * mu_bb
        df["energy_density_mevfm3"] = df["energy_density"] * GEV4_TO_MEV_FM3
        df["pressure_mevfm3"] = df["pressure"] * GEV4_TO_MEV_FM3
        return df


def build_quarkyonic_eos_objects(
    fit_df: pd.DataFrame,
    target_k0_values: Iterable[float] | None = None,
    **eos_kwargs,
) -> dict[str, AsymmetricClausiusQuarkyonicEOS]:
    """Build quarkyonic-EOS objects from the Clausius fit dataframe."""
    parameter_dict = fit_df_to_parameter_dict(fit_df, target_k0_values=target_k0_values)
    return {
        label: AsymmetricClausiusQuarkyonicEOS(label, params, **eos_kwargs)
        for label, params in parameter_dict.items()
    }


def save_fixed_y_tables(
    profile_map: dict[float, pd.DataFrame],
    output_dir: str | Path,
    label: str,
) -> list[Path]:
    """Save `energy_density`/`pressure` tables for one EOS branch."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for y, df in profile_map.items():
        y_str = f"{y:.1f}"
        fname = output_path / f"energy_vs_pressure_clausius_{label}_y{y_str}.dat"
        data = np.column_stack((df["energy_density_mevfm3"], df["pressure_mevfm3"]))
        np.savetxt(
            fname,
            data,
            header="MeV/fm^3               MeV/fm^3\nenergy_density         pressure",
            fmt="%.10e",
            comments="# ",
        )
        written.append(fname)
    return written
