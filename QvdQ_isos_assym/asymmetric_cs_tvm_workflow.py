import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root, root_scalar

from src.numerical_functions import integrate_simpson


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
GEV_TO_MEV = 1000.0
GEV3_TO_FM3 = (1.0 / 0.1973269804) ** 3
GEV4_TO_MEV_FM3 = 1.30148926289e5

DEFAULT_TARGET_S0 = 32.5
DEFAULT_TARGET_L0 = 58.9

PUBLISHED_PARAMETERS = {
    "cs": {
        "eq": {"an": 36.93, "bn": 579.17, "apn": 53.60, "bpn": 579.17},
        "neq": {"an": 26.535, "bn": 383.788, "apn": 63.999, "bpn": 774.64},
    },
    "tvm": {
        "eq": {"an": 37.33, "bn": 560.08, "apn": 53.72, "bpn": 560.08},
        "neq": {"an": 26.78, "bn": 363.72, "apn": 64.67, "bpn": 756.4},
    },
}


def symmetric_baselines():
    return {
        model: {
            "a_bar": 0.5 * (branches["eq"]["an"] + branches["eq"]["apn"]),
            "b_bar": 0.5 * (branches["eq"]["bn"] + branches["eq"]["bpn"]),
        }
        for model, branches in PUBLISHED_PARAMETERS.items()
    }


def default_density_grid(n_points=51, low=0.75, high=1.25):
    return np.linspace(low, high, n_points) * RHO0


def default_rhob_list(n_points=120, low=1.0e-6, high=5.0):
    return np.linspace(low, high, n_points) * RHO0


def default_y_values():
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def f_tvm(x):
    return np.exp(-x - 0.5 * x * x)


def f_cs(x):
    x_arr = np.asarray(x, dtype=float)
    out = np.exp(-(3.0 * x_arr) / (4.0 - x_arr) - (4.0 * x_arr) / (4.0 - x_arr) ** 2)
    out = np.where(x_arr < 4.0, out, np.nan)
    return out if out.ndim else float(out)


def volume_fraction(model, x):
    if model == "cs":
        return f_cs(x)
    if model == "tvm":
        return f_tvm(x)
    raise ValueError(f"Unknown model: {model}")


def branch_parameters(a_bar, b_bar, delta_a, delta_b):
    return {
        "an": a_bar - delta_a,
        "apn": a_bar + delta_a,
        "bn": b_bar - delta_b,
        "bpn": b_bar + delta_b,
    }


def kf_from_nid(n_id, g=G):
    return (6.0 * np.pi**2 * n_id / g) ** (1.0 / 3.0)


def eps_id_fermion(mass, kf, g=G):
    ef = np.sqrt(kf * kf + mass * mass)
    return g / (16.0 * np.pi**2) * (
        kf * ef * (2.0 * kf * kf + mass * mass) - mass**4 * np.log((kf + ef) / mass)
    )


def energy_per_baryon_hadronic(n_b, y, *, model, an, bn, apn, bpn, mp=MP, mn=MN):
    n_p = n_b * y
    n_n = n_b * (1.0 - y)

    x_p = bn * n_p + bpn * n_n
    x_n = bpn * n_p + bn * n_n

    f_p = volume_fraction(model, x_p)
    f_n = volume_fraction(model, x_n)

    if (not np.isfinite(f_p)) or (not np.isfinite(f_n)) or f_p <= 0.0 or f_n <= 0.0:
        return np.nan

    kfp = kf_from_nid(n_p / f_p) if n_p > 0.0 else 0.0
    kfn = kf_from_nid(n_n / f_n) if n_n > 0.0 else 0.0

    eps = (
        f_p * eps_id_fermion(mp, kfp)
        + f_n * eps_id_fermion(mn, kfn)
        - an * (n_p**2 + n_n**2)
        - 2.0 * apn * n_p * n_n
    )
    return eps / n_b


def symmetry_curves(model, params, density_grid):
    e_snm = []
    e_pnm = []
    for n_b in density_grid:
        e_snm.append((energy_per_baryon_hadronic(n_b, 0.5, model=model, **params) - MP) * GEV_TO_MEV)
        e_pnm.append((energy_per_baryon_hadronic(n_b, 0.0, model=model, **params) - MP) * GEV_TO_MEV)

    e_snm = np.asarray(e_snm, dtype=float)
    e_pnm = np.asarray(e_pnm, dtype=float)
    s_vals = e_pnm - e_snm
    n_fm = density_grid * GEV3_TO_FM3
    l_vals = 3.0 * n_fm * np.gradient(s_vals, n_fm)
    return s_vals, l_vals


def s0_l0(model, a_bar, b_bar, delta_a, delta_b, density_grid):
    params = branch_parameters(a_bar, b_bar, delta_a, delta_b)
    s_vals, l_vals = symmetry_curves(model, params, density_grid)
    idx = np.argmin(np.abs(density_grid - RHO0))
    return float(s_vals[idx]), float(l_vals[idx]), params


def fit_eq_branch(model, a_bar, b_bar, density_grid, target_s0=DEFAULT_TARGET_S0):
    def objective(delta_a):
        s0, _, _ = s0_l0(model, a_bar, b_bar, delta_a, 0.0, density_grid)
        return s0 - target_s0

    sol = root_scalar(objective, bracket=(-30.0, 30.0), method="brentq")
    s0, l0, params = s0_l0(model, a_bar, b_bar, sol.root, 0.0, density_grid)
    return {
        "model": model,
        "branch": "eq",
        "delta_a": float(sol.root),
        "delta_b": 0.0,
        "S0": s0,
        "L0": l0,
        **params,
    }


def fit_neq_branch(
    model,
    a_bar,
    b_bar,
    density_grid,
    target_s0=DEFAULT_TARGET_S0,
    target_l0=DEFAULT_TARGET_L0,
):
    def objective(x):
        delta_a, delta_b = x
        s0, l0, _ = s0_l0(model, a_bar, b_bar, delta_a, delta_b, density_grid)
        return np.array([s0 - target_s0, l0 - target_l0])

    sol = root(objective, x0=np.array([18.0, 190.0]), method="hybr")
    if not sol.success:
        raise RuntimeError(f"Root solve failed for {model}: {sol.message}")

    delta_a, delta_b = sol.x
    s0, l0, params = s0_l0(model, a_bar, b_bar, delta_a, delta_b, density_grid)
    return {
        "model": model,
        "branch": "neq",
        "delta_a": float(delta_a),
        "delta_b": float(delta_b),
        "S0": s0,
        "L0": l0,
        **params,
    }


def fit_parameter_table(
    target_s0=DEFAULT_TARGET_S0,
    target_l0=DEFAULT_TARGET_L0,
    density_grid=None,
):
    if density_grid is None:
        density_grid = default_density_grid()

    fit_rows = []
    for model, base in symmetric_baselines().items():
        fit_rows.append(fit_eq_branch(model, base["a_bar"], base["b_bar"], density_grid, target_s0=target_s0))
        fit_rows.append(
            fit_neq_branch(
                model,
                base["a_bar"],
                base["b_bar"],
                density_grid,
                target_s0=target_s0,
                target_l0=target_l0,
            )
        )
    return pd.DataFrame(fit_rows)


def fitted_parameters_dict(fit_df):
    return {
        model: {
            branch: {
                "an": float(row.an),
                "bn": float(row.bn),
                "apn": float(row.apn),
                "bpn": float(row.bpn),
            }
            for branch, row in fit_df[fit_df.model == model].set_index("branch").iterrows()
        }
        for model in fit_df.model.unique()
    }


def published_comparison(fit_df, density_grid=None):
    if density_grid is None:
        density_grid = default_density_grid()

    rows = []
    for model, branches in PUBLISHED_PARAMETERS.items():
        for branch, params in branches.items():
            s0, l0, _ = s0_l0(
                model,
                0.5 * (params["an"] + params["apn"]),
                0.5 * (params["bn"] + params["bpn"]),
                0.5 * (params["apn"] - params["an"]),
                0.5 * (params["bpn"] - params["bn"]),
                density_grid,
            )
            rows.append(
                {
                    "model": model,
                    "branch": branch,
                    "published_S0": s0,
                    "published_L0": l0,
                    **params,
                }
            )

    published_df = pd.DataFrame(rows)
    comparison = fit_df.merge(published_df, on=["model", "branch"], suffixes=("_fit", "_pub"))
    for key in ["an", "bn", "apn", "bpn"]:
        comparison[f"d_{key}"] = comparison[f"{key}_fit"] - comparison[f"{key}_pub"]
    return comparison


def gss_safe(func, a, b, tol=1.0e-6, max_iter=400):
    gr = (math.sqrt(5.0) + 1.0) / 2.0
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


class AsymmetricQuarkyonicEOS:
    def __init__(
        self,
        model,
        params,
        *,
        g=G,
        nc=NC,
        mp=MP,
        mn=MN,
        me=ME,
        mmu=MMU,
        mu=MU,
        md=MD,
        lam=LAM,
        rho0=RHO0,
        nint=300,
    ):
        self.model = model
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
    def an(self):
        return self.params["an"]

    @property
    def bn(self):
        return self.params["bn"]

    @property
    def apn(self):
        return self.params["apn"]

    @property
    def bpn(self):
        return self.params["bpn"]

    def euid(self, kbu, y, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(q):
            return q * np.sqrt(self.lam**2 + q**2) * np.sqrt(self.mu**2 + q**2)

        result = (1.0 + y) / (2.0 - y) * integrate_simpson(f, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def edid(self, kbu, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(q):
            return q * np.sqrt(self.lam**2 + q**2) * np.sqrt(self.md**2 + q**2)

        result = integrate_simpson(f, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def enid(self, kbu, kfn, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(k):
            return k**2 * np.sqrt(self.mn**2 + k**2)

        result = integrate_simpson(f, kbu, kfn, n=nint)
        return pref * result

    def epid(self, kbu, kfp, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(k):
            return k**2 * np.sqrt(self.mp**2 + k**2)

        result = integrate_simpson(f, kbu, kfp, n=nint)
        return pref * result

    def nuid(self, kbu, y, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(q):
            return q * np.sqrt(self.lam**2 + q**2)

        result = (1.0 + y) / (2.0 - y) * integrate_simpson(f, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def ndid(self, kbu, nint=None):
        if nint is None:
            nint = self.nint
        pref = self.g / (2.0 * np.pi**2)

        def f(q):
            return q * np.sqrt(self.lam**2 + q**2)

        result = integrate_simpson(f, 0.0, kbu / self.nc, n=nint)
        return self.nc * pref * result

    def kbu_solver(self, rhob, fq, y):
        pref = self.g / (2.0 * np.pi**2)
        w = (3.0 / (2.0 - y)) * pref
        nq = rhob * fq
        inside = (3.0 * nq / w + self.lam**3) ** (2.0 / 3.0) - self.lam**2
        return self.nc * np.sqrt(max(inside, 0.0))

    def _hadronic_densities(self, rhob, fq, y):
        n_p = rhob * (1.0 - fq) * y
        n_n = rhob * (1.0 - fq) * (1.0 - y)
        return n_p, n_n

    def _x_p_x_n(self, rhob, fq, y):
        n_p, n_n = self._hadronic_densities(rhob, fq, y)
        x_p = self.bn * n_p + self.bpn * n_n
        x_n = self.bpn * n_p + self.bn * n_n
        return x_p, x_n

    def kfp_solver(self, rhob, fq, y, kbu):
        n_p, n_n = self._hadronic_densities(rhob, fq, y)
        x_p = self.bn * n_p + self.bpn * n_n
        f_p = volume_fraction(self.model, x_p)
        if not np.isfinite(f_p) or f_p <= 0.0:
            return np.nan
        n_p_id = n_p / f_p
        val = kbu**3 + (6.0 * np.pi**2 / self.g) * n_p_id
        return np.cbrt(max(val, 0.0))

    def kfn_solver(self, rhob, fq, y, kbu):
        n_p, n_n = self._hadronic_densities(rhob, fq, y)
        x_n = self.bpn * n_p + self.bn * n_n
        f_n = volume_fraction(self.model, x_n)
        if not np.isfinite(f_n) or f_n <= 0.0:
            return np.nan
        n_n_id = n_n / f_n
        val = kbu**3 + (6.0 * np.pi**2 / self.g) * n_n_id
        return np.cbrt(max(val, 0.0))

    def energy_hq_of_fq_y(self, fq, rhob, y):
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
        x_p, x_n = self._x_p_x_n(rhob, fq, y)

        f_p = volume_fraction(self.model, x_p)
        f_n = volume_fraction(self.model, x_n)
        if (not np.isfinite(f_p)) or (not np.isfinite(f_n)) or f_p <= 0.0 or f_n <= 0.0:
            return np.nan

        nuclear_int = f_p * ep + f_n * en - self.an * (n_p**2 + n_n**2) - 2.0 * self.apn * n_p * n_n
        return eu + ed + nuclear_int

    def lepton_kf_from_mu(self, mu_l, m_l):
        return np.sqrt(max(mu_l**2 - m_l**2, 0.0))

    def lepton_density_from_mu(self, mu_l, m_l):
        if mu_l <= m_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, m_l)
        return kf**3 / (3.0 * np.pi**2)

    def lepton_energy_density_from_mu(self, mu_l, m_l, nint=None):
        if nint is None:
            nint = self.nint
        if mu_l <= m_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, m_l)

        def f(k):
            return k**2 * np.sqrt(k**2 + m_l**2)

        return (1.0 / np.pi**2) * integrate_simpson(f, 0.0, kf, n=nint)

    def charge_density_hq(self, rhob, fq, y):
        kbu = self.kbu_solver(rhob, fq, y)
        if not np.isfinite(kbu):
            return np.nan

        n_p = rhob * (1.0 - fq) * y
        n_u = 3.0 * self.nuid(kbu, y)
        n_d = 3.0 * self.ndid(kbu)
        return n_p + (2.0 / 3.0) * n_u - (1.0 / 3.0) * n_d

    def mu_q_fd(self, rhob, fq, y, dy=1.0e-5):
        y1 = max(1.0e-8, y - dy)
        y2 = min(0.5 - 1.0e-8, y + dy)
        e1 = self.energy_hq_of_fq_y(fq, rhob, y1)
        e2 = self.energy_hq_of_fq_y(fq, rhob, y2)
        if (not np.isfinite(e1)) or (not np.isfinite(e2)):
            return np.nan
        return (e2 - e1) / (rhob * (y2 - y1))

    def mu_e_from_y(self, rhob, fq, y):
        mu_q = self.mu_q_fd(rhob, fq, y)
        if not np.isfinite(mu_q):
            return np.nan
        return -mu_q

    def neutrality_residual(self, y, rhob, fq):
        mu_e = self.mu_e_from_y(rhob, fq, y)
        if not np.isfinite(mu_e):
            return np.nan

        rho_hq = self.charge_density_hq(rhob, fq, y)
        rho_e = self.lepton_density_from_mu(mu_e, self.me)
        rho_mu = self.lepton_density_from_mu(mu_e, self.mmu)
        if not np.isfinite(rho_hq):
            return np.nan
        return rho_hq - (rho_e + rho_mu)

    def solve_y_beta_equilibrium(self, rhob, fq, y_min=1.0e-6, y_max=0.5 - 1.0e-6, nscan=160):
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

    def total_energy_beta(self, fq, rhob, y_scan=160):
        y_star, _, eps_e, eps_mu = self.solve_y_beta_equilibrium(rhob, fq, nscan=y_scan)
        if not np.isfinite(y_star):
            return np.nan
        eps_hq = self.energy_hq_of_fq_y(fq, rhob, y_star)
        if not np.isfinite(eps_hq):
            return np.nan
        return eps_hq + eps_e + eps_mu

    def solve_fq_fixed_y(self, rhob, y, fq_min=1.0e-3, fq_max=1.0, tol=1.0e-6):
        return gss_safe(lambda fq: self.energy_hq_of_fq_y(fq, rhob, y), fq_min, fq_max, tol=tol)

    def solve_fq_beta(self, rhob, fq_min=1.0e-3, fq_max=1.0, tol=1.0e-6, y_scan=160):
        return gss_safe(lambda fq: self.total_energy_beta(fq, rhob, y_scan=y_scan), fq_min, fq_max, tol=tol)

    def beta_equilibrium_profile(self, rhob_list, fq_min=1.0e-3, fq_max=1.0, tol=1.0e-6, y_scan=160):
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
                }
            )

        df = pd.DataFrame(rows)
        e = df["energy_density"].to_numpy(dtype=float)
        n = df["rhob"].to_numpy(dtype=float)
        mu_b = np.gradient(e, n, edge_order=2)
        mu_bb = np.gradient(mu_b, n, edge_order=2)
        pressure = n * mu_b - e

        df["mu_b"] = mu_b
        df["pressure"] = pressure
        df["vs2"] = n / mu_b * mu_bb
        df["energy_density_mevfm3"] = df["energy_density"] * GEV4_TO_MEV_FM3
        df["pressure_mevfm3"] = df["pressure"] * GEV4_TO_MEV_FM3
        return df

    def fixed_y_profile(self, rhob_list, y, fq_min=1.0e-3, fq_max=1.0, tol=1.0e-6):
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
                }
            )

        df = pd.DataFrame(rows)
        e = df["energy_density"].to_numpy(dtype=float)
        n = df["rhob"].to_numpy(dtype=float)
        mu_b = np.gradient(e, n, edge_order=2)
        mu_bb = np.gradient(mu_b, n, edge_order=2)
        pressure = n * mu_b - e

        df["mu_b"] = mu_b
        df["pressure"] = pressure
        df["vs2"] = n / mu_b * mu_bb
        df["energy_density_mevfm3"] = df["energy_density"] * GEV4_TO_MEV_FM3
        df["pressure_mevfm3"] = df["pressure"] * GEV4_TO_MEV_FM3
        return df


def build_eos_objects(parameters_dict, **eos_kwargs):
    return {
        model: {
            branch: AsymmetricQuarkyonicEOS(model, params, **eos_kwargs)
            for branch, params in branches.items()
        }
        for model, branches in parameters_dict.items()
    }


def save_fixed_y_tables(profile_map, output_dir, model, branch):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = []
    for y, df in profile_map.items():
        y_str = f"{y:.1f}"
        fname = output_path / f"energy_vs_pressure_{model}_{branch}_y{y_str}.dat"
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
