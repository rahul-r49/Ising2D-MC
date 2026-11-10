"""Finite-size scaling and growth-law analysis used for the dissertation results.

All functions work on plain NumPy arrays. Temperatures are in units of J / k_B.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .temperatures import grid_matching

# Onsager's exact critical temperature of the 2D square-lattice Ising model
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))      # 2.26918...
# Exact critical exponents of the 2D Ising universality class
GAMMA_OVER_NU = 1.75
NU = 1.0
BETA = 0.125


# ---------------------------------------------------------------- loading data
def load_observable(data_dir, name, L):
    """Seed-averaged observable of lattice size L, sorted by increasing temperature.

    name is one of 'U' (Binder cumulant), 'Sus' (susceptibility), 'Abs_m' (<|m|>)
    or 'm' (<m>). The file is <data_dir>/<name>_<L>.csv, one value per temperature
    in the order the annealing visited them (highest T first).
    Returns (T, values).
    """
    values = pd.read_csv(Path(data_dir) / f"{name}_{L}.csv").iloc[:, 0].to_numpy()
    T = grid_matching(len(values))
    order = np.argsort(T)
    return T[order], values[order]


# --------------------------------------------------------------------- fitting
def linear_fit(x, y):
    """Straight-line fit y = a*x + b. Returns (a, b, err_a, err_b), where the errors
    are the standard errors of the fit (they do not include the statistical error of y)."""
    (a, b), cov = np.polyfit(x, y, 1, cov=True)
    return a, b, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def loglog_slope(x, y):
    """Slope of ln y against ln x with its standard error."""
    a, _, err_a, _ = linear_fit(np.log(x), np.log(y))
    return a, err_a


def chi_max(T, chi):
    """Temperature and value of the susceptibility peak."""
    i = int(np.argmax(chi))
    return T[i], chi[i]


def extrapolate_tc(L, T_star, nu=NU):
    """Critical temperature in the thermodynamic limit from the pseudo-critical
    temperatures T*(L) of finite lattices:

        T*(L) = Tc + b L^(-1/nu)

    A straight-line fit of T*(L) against L^(-1/nu); the intercept is Tc.
    Returns (Tc, b, err_Tc).
    """
    x = np.asarray(L, dtype=float) ** (-1.0 / nu)
    b, Tc, _, err_Tc = linear_fit(x, T_star)
    return Tc, b, err_Tc


def collapse_susceptibility(T, chi, L, Tc=TC_EXACT, nu=NU, gamma_over_nu=GAMMA_OVER_NU):
    """Finite-size scaling variables  y = (T - Tc) L^(1/nu)  and  Y = L^(-gamma/nu) chi.
    Curves of different L fall on one curve when Tc and the exponents are right."""
    return (T - Tc) * L ** (1.0 / nu), chi * L ** (-gamma_over_nu)


def collapse_binder(T, U, L, Tc=TC_EXACT, nu=NU):
    """x = (T - Tc) L^(1/nu); U4 itself needs no rescaling because it is dimensionless."""
    return (T - Tc) * L ** (1.0 / nu), U


# --------------------------------------------------------- phase-ordering growth
def running_average(a, window):
    """Moving average over `window` points (shorter by window - 1)."""
    return np.convolve(a, np.ones(window) / window, mode="valid")


def instantaneous_exponent(t, R, window=50):
    """alpha_i = d ln l / d ln t, after smoothing ln l and ln t with a moving average.
    Returns (1 / l, alpha_i) so that alpha_i can be plotted against 1 / l(t)."""
    ln_R = running_average(np.log(R), window)
    ln_t = running_average(np.log(t), window)
    return 1.0 / np.exp(ln_R), np.gradient(ln_R, ln_t)


def growth_exponent(inv_l, alpha_i, lo=0.022, hi=0.056):
    """Growth exponent alpha: straight-line fit of alpha_i against 1/l(t) in the window
    lo < 1/l < hi, extrapolated to 1/l -> 0 (infinite system). Returns (alpha, err)."""
    m = (inv_l > lo) & (inv_l < hi)
    _, intercept, _, err = linear_fit(inv_l[m], alpha_i[m])
    return intercept, err


def kinetics_collapse(t, R, l0, t0, alpha=0.5):
    """Finite-size scaling of the domain growth for a lattice with maximum domain size
    l_max = max(R):

        y = (l_max - l0)^(1/alpha) / (t - t0),    Y = (l(t) - l0) / (l_max - l0)

    Curves of different L collapse for the right alpha, l0 and t0.
    The point t = t0 is dropped (y diverges there).
    """
    t = np.asarray(t, dtype=float)
    keep = t != t0
    l_max = np.max(R)
    y = (l_max - l0) ** (1.0 / alpha) / (t[keep] - t0)
    Y = (np.asarray(R)[keep] - l0) / (l_max - l0)
    return y, Y


# ------------------------------------------------------------------- summary
def summarise(data_dir):
    """All headline numbers from a folder of seed-averaged results (see README).

    Needs Sus_<L>.csv and U_<L>.csv files. The m = 0 pseudo-critical temperatures
    (pseudo_critical_m0.csv) and the domain-growth files R_<L>_mean.csv are used
    when they are present. Returns a dict.
    """
    from .coarsening import measurement_times

    data_dir = Path(data_dir)
    Ls = sorted(int(p.stem.split("_")[1]) for p in data_dir.glob("Sus_*.csv"))
    if len(Ls) < 3:
        raise ValueError(f"need Sus_<L>.csv and U_<L>.csv for at least 3 lattice sizes "
                         f"in {data_dir} (found {len(Ls)})")
    out = {"Ls": Ls}

    # susceptibility peak -> Tc by extrapolation and gamma/nu from chi_max ~ L^(gamma/nu)
    peaks = [chi_max(*load_observable(data_dir, "Sus", L)) for L in Ls]
    T_star = np.array([p[0] for p in peaks])
    chi_peak = np.array([p[1] for p in peaks])
    out["T_star_chi"] = T_star
    out["chi_max"] = chi_peak
    out["Tc_chi"], _, out["Tc_chi_err"] = extrapolate_tc(Ls, T_star)
    out["gamma_over_nu"], out["gamma_over_nu_err"] = loglog_slope(Ls, chi_peak)

    # m = 0 pseudo-critical temperatures (read from the distribution histograms)
    f = data_dir / "pseudo_critical_m0.csv"
    if f.exists():
        t = pd.read_csv(f)
        out["Tc_m0"], _, out["Tc_m0_err"] = extrapolate_tc(t["L"], t["T_star"])

    # domain growth: instantaneous exponent of the largest lattice
    R = {int(p.stem.split("_")[1]): pd.read_csv(p).iloc[:, 0].to_numpy()
         for p in data_dir.glob("R_*_mean.csv")}
    if R:
        times = measurement_times()
        L_big = max(R)
        out["growth_L"] = L_big
        inv_l, alpha_i = instantaneous_exponent(times, R[L_big])
        out["alpha"], _ = growth_exponent(inv_l, alpha_i)
    return out
