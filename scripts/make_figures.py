"""Make the figures from a folder of seed-averaged results.

    python scripts/make_figures.py data/dissertation

The figures are saved in results/<folder name>/ (here results/dissertation/), so
figures made from your own data do not overwrite the ones made from the stored
results. A text file with the headline numbers (summary.txt) is saved with them.

The folder needs  Sus_<L>.csv  and  U_<L>.csv  for each lattice size L, and
Abs_m_<L>.csv  for the magnetisation plot. The phase diagram, the m = 0 estimate of
Tc and the domain growth plots are made when  phase_diagram_peaks.csv,
pseudo_critical_m0.csv  and  R_<L>_mean.csv  are present. Domain growth data is
assumed to be measured at the times given by ising.coarsening.measurement_times()
with its default settings (20000 sweeps).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ising import analysis as A
from ising.coarsening import measurement_times

plt.rcParams.update({
    "xtick.top": True, "ytick.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.dpi": 100,
})

TC = A.TC_EXACT
MARKERS = ["^", "s", "p", "P", "h", "v"]
COLOURS = ["tab:blue", "tab:orange", "black", "tab:green", "tab:red", "tab:brown"]


def style(i):
    return dict(marker=MARKERS[i % 6], color=COLOURS[i % 6])


# ------------------------------------------------------------------ panels
def binder(ax, data, Ls, summary):
    for i, L in enumerate(Ls):
        T, U = A.load_observable(data, "U", L)
        ax.plot(T, U, ls=":", markerfacecolor="none", ms=5, label=f"L = {L}", **style(i))
    # the Binder curves cross at T_c; the line marks the T_c found from the susceptibility peaks
    ax.axvline(summary["Tc_chi"], color="k", ls="--", lw=1)
    ax.text(summary["Tc_chi"] - 0.005, 0.12, f"$T_c$ (from $\\chi_{{\\max}}$)\n{summary['Tc_chi']:.3f}", ha="right", fontsize=9)
    ax.axvline(TC, color="r", ls=":", lw=1)
    ax.text(TC + 0.005, 0.12, f"exact\n{TC:.3f}", ha="left", fontsize=9, color="r")
    ax.set_xlim(2.1, 2.5)
    ax.set_xlabel(r"Temperature, $T$ ($J/k_B$)")
    ax.set_ylabel(r"Binder cumulant $U_4$")
    ax.legend(frameon=False, loc="lower left")


def susceptibility(ax, data, Ls):
    for i, L in enumerate(Ls):
        T, chi = A.load_observable(data, "Sus", L)
        ax.plot(T, chi, ls="none", markerfacecolor="none", ms=5, label=f"L = {L}", **style(i))
    ax.axvline(TC, color="r", ls=":", lw=1)
    ax.set_xlim(1.6, 3.0)
    ax.set_xlabel(r"Temperature, $T$ ($J/k_B$)")
    ax.set_ylabel(r"Susceptibility, $\chi$")
    ax.legend(frameon=False, loc="upper left")


def susceptibility_collapse(ax, data, Ls):
    for i, L in enumerate(Ls):
        T, chi = A.load_observable(data, "Sus", L)
        y, Y = A.collapse_susceptibility(T, chi, L)
        keep = y > 0                                   # only T > Tc fits on a log axis
        ax.loglog(y[keep], Y[keep], ls="none", markerfacecolor="none", ms=5, label=f"L = {L}", **style(i))
    x = np.array([10.0, 80.0])
    ax.loglog(x, 0.4 * x ** -1.75, "m--", lw=1)
    ax.text(25, 1.2e-2, r"slope $-1.75$", color="m")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-4, 1e-1)
    ax.set_xlabel(r"$y = (T - T_c)\,L^{1/\nu}$   ($T > T_c$)")
    ax.set_ylabel(r"$Y(y) = L^{-\gamma/\nu}\chi$")
    ax.legend(frameon=False, loc="lower left")


def chi_max_scaling(ax, summary):
    L, chi = np.array(summary["Ls"]), summary["chi_max"]
    ax.loglog(L, chi, "o", ls=":")
    s = summary["gamma_over_nu"]
    fit = np.exp(np.polyval(np.polyfit(np.log(L), np.log(chi), 1), np.log(L)))
    ax.loglog(L, fit, "r-", lw=1)
    ax.text(0.05, 0.9, rf"slope $\gamma/\nu$ = {s:.3f}",
            transform=ax.transAxes)
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\chi_{\max}$")


def tc_extrapolation(ax, L, T_star, label, offset=0.0):
    Tc, b, _ = A.extrapolate_tc(L, T_star)
    x = 1.0 / np.asarray(L, dtype=float)
    ax.plot(x, T_star, "o")
    xx = np.linspace(0, x.max(), 50)
    ax.plot(xx, Tc + b * xx, "--", color="orange")
    ax.plot([0], [Tc], "k<", ms=6, clip_on=False, zorder=5)
    ax.axhline(TC, color="r", ls=":", lw=1)
    ax.text(0.03, 0.95, f"{label}: $T_c$ = {Tc:.3f}\nexact: {TC:.4f}",
            transform=ax.transAxes, ha="left", va="top")
    ax.set_xlim(0, None)             # the fitted line meets the y axis at x = 1/L = 0
    ax.set_xlabel("1 / L")
    ax.set_ylabel(r"$T_c^{*}(L)$")


def phase_diagram(ax, data):
    p = pd.read_csv(Path(data) / "phase_diagram_peaks.csv")
    for i, (L, g) in enumerate(p.groupby("L")):
        ax.plot(g["m_negative_peak"], g["T"], ls="none", ms=5, label=f"L = {L}", **style(i))
        ax.plot(g["m_positive_peak"], g["T"], ls="none", ms=5, **style(i))
    ax.axvline(0, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Magnetisation per spin, m")
    ax.set_ylabel(r"Temperature, $T$ ($J/k_B$)")
    ax.legend(frameon=False, loc="lower center", ncol=2)


def critical_exponents(axes, data, L):
    T, m = A.load_observable(data, "Abs_m", L)
    _, chi = A.load_observable(data, "Sus", L)
    below = (TC - T) > 0.01
    ax = axes[0]
    ax.loglog((TC - T)[below], m[below], "o", ms=4)
    x = np.array([0.01, 0.2])
    ref = np.interp(0.05, (TC - T)[below][::-1], m[below][::-1])
    ax.loglog(x, ref * (x / 0.05) ** 0.125, "r--", lw=1)
    ax.text(0.03, 0.55, r"slope $\beta = 0.125$", color="r", transform=ax.transAxes)
    ax.set_xlabel(r"$|T - T_c|$")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_ylim(0.5, 1)
    ax.set_title(f"L = {L}, T < $T_c$")
    ax = axes[1]
    up, dn = (T - TC) > 0.01, (TC - T) > 0.01
    ax.loglog((T - TC)[up], chi[up], "o", ms=4, label=r"$T > T_c$")
    ax.loglog((TC - T)[dn], chi[dn], "s", ms=4, label=r"$T < T_c$")
    x = np.array([0.03, 0.4])
    for sel, c in ((up, "C0"), (dn, "C1")):
        t = np.abs(T - TC)[sel]
        v = chi[sel]
        k = np.argmin(abs(t - 0.1))
        ax.loglog(x, v[k] * (x / t[k]) ** -1.75, "--", color=c, lw=1)
    ax.text(0.5, 0.85, r"dashed: slope $-\gamma = -1.75$", transform=ax.transAxes)
    ax.set_xlim(0.02, 1)             # zoom on the region used for the exponent
    ax.set_ylim(2, 200)
    ax.set_xlabel(r"$|T - T_c|$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(f"L = {L}")
    ax.legend(frameon=False, loc="lower left")


def load_growth(data):
    t = measurement_times()
    R = {}
    for p in sorted(Path(data).glob("R_*_mean.csv")):
        L = int(p.stem.split("_")[1])
        r = pd.read_csv(p).iloc[:, 0].to_numpy()
        if len(r) != len(t):
            raise SystemExit(f"{p.name} has {len(r)} values but the default measurement times "
                             f"have {len(t)}; the domain growth plots assume 20000 sweeps")
        R[L] = r
    return t, R


def growth(ax, t, R):
    for i, (L, r) in enumerate(R.items()):
        ax.loglog(t, r, "o", ms=4, markerfacecolor="none", label=f"L = {L}", color=COLOURS[i])
    x = np.array([10.0, 1000.0])
    ax.loglog(x, 4.4 * x ** 0.5, "k--", lw=1)
    ax.text(12, 2.5e2, r"slope 0.5")
    ax.set_xlabel("t (MCS)")
    ax.set_ylabel(r"$\ell(t)$")
    ax.legend(frameon=False, loc="upper left")


def alpha_extrapolation(ax, t, R):
    L = max(R)
    inv_l, alpha_i = A.instantaneous_exponent(t, R[L])
    lo, hi = 0.022, 0.056
    m = (inv_l > lo) & (inv_l < hi)
    slope, intercept, _, _ = A.linear_fit(inv_l[m], alpha_i[m])
    ax.plot(inv_l, alpha_i, "o", ms=3, markerfacecolor="none", color="blue", label=f"L = {L}")
    xx = np.linspace(0, hi, 50)
    ax.plot(xx, slope * xx + intercept, "k--", lw=1.2)
    ax.plot([0], [intercept], "k<", ms=6)
    ax.text(0.004, intercept + 0.002, f"{intercept:.3f}")
    ax.set_xlim(0, 0.07)
    ax.set_ylim(0.48, 0.53)
    ax.set_xlabel(r"$1/\ell(t)$")
    ax.set_ylabel(r"$\alpha_i$")
    ax.legend(frameon=False, loc="upper right")


def kinetics_collapse(ax, t, R):
    l0, t0 = next(iter(R.values()))[0], t[0]
    for i, (L, r) in enumerate(R.items()):
        y, Y = A.kinetics_collapse(t, r, l0, t0)
        ax.loglog(y, Y, ls="none", marker=MARKERS[i], ms=5, markerfacecolor="none", color=COLOURS[i], label=f"L = {L}")
    x = np.array([50.0, 1000.0])
    ax.loglog(x, 4.5 * x ** -0.5, "r--", lw=1)
    ax.text(300, 0.3, r"slope $-0.5$", color="r")
    ax.set_ylim(0.09, 1.1)
    ax.set_xlim(None, 1e4)
    ax.set_xlabel("y")
    ax.set_ylabel("Y(y)")
    ax.legend(frameon=False, loc="lower left")


# ------------------------------------------------------------------ driver
def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


def main(data):
    data = Path(data)
    out = Path("results") / data.name
    out.mkdir(parents=True, exist_ok=True)

    s = A.summarise(data)
    Ls = s["Ls"]
    has_phase = (data / "phase_diagram_peaks.csv").exists() and "Tc_m0" in s
    has_growth = any(data.glob("R_*_mean.csv"))
    has_m = (data / f"Abs_m_{max(Ls)}.csv").exists()

    fig, ax = plt.subplots(figsize=(6, 5)); binder(ax, data, Ls, s); save(fig, out, "binder_cumulant.png")
    fig, ax = plt.subplots(figsize=(6, 5)); susceptibility(ax, data, Ls); save(fig, out, "susceptibility.png")
    fig, ax = plt.subplots(figsize=(6, 5)); susceptibility_collapse(ax, data, Ls); save(fig, out, "susceptibility_collapse.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    chi_max_scaling(ax[0], s)
    tc_extrapolation(ax[1], Ls, s["T_star_chi"], r"from $\chi_{\max}$")
    save(fig, out, "fss_exponents.png")

    if has_phase:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        phase_diagram(ax[0], data)
        t = pd.read_csv(data / "pseudo_critical_m0.csv")
        tc_extrapolation(ax[1], t["L"], t["T_star"], r"from $m = 0$")
        save(fig, out, "phase_diagram.png")

    if has_m:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        critical_exponents(ax, data, max(Ls))
        save(fig, out, "critical_exponents.png")

    if has_growth:
        t, R = load_growth(data)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        growth(ax[0], t, R)
        alpha_extrapolation(ax[1], t, R)
        kinetics_collapse(ax[2], t, R)
        save(fig, out, "domain_growth.png")

    # overview: one panel from each part of the study
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    binder(ax[0, 0], data, Ls, s)
    susceptibility_collapse(ax[0, 1], data, Ls)
    tc_extrapolation(ax[1, 0], Ls, s["T_star_chi"], r"from $\chi_{\max}$")
    if has_growth:
        growth(ax[1, 1], *load_growth(data))
    else:
        ax[1, 1].axis("off")
    for a, lab in zip(ax.ravel(), "abcd"):
        a.set_title(f"({lab})", loc="left", fontweight="bold")
    save(fig, out, "overview.png")

    lines = [f"lattice sizes: {Ls}",
             f"Tc from chi_max extrapolation : {s['Tc_chi']:.3f}",
             f"gamma/nu from chi_max         : {s['gamma_over_nu']:.3f}",
             f"exact Tc (Onsager)            : {TC:.4f}"]
    if "Tc_m0" in s:
        lines.insert(2, f"Tc from m = 0                 : {s['Tc_m0']:.3f}")
    if "alpha" in s:
        lines.append(f"growth exponent alpha (L={s['growth_L']}) : {s['alpha']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
