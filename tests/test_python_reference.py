"""Tests of the Numba reference code (src/ising/metropolis.py)."""
import numpy as np

import exact
from ising import anneal, observables


def test_matches_exact_4x4_lattice():
    temps = np.array([4.5, 3.0, 2.5, 2.0, 1.5])
    n = len(temps)
    eq, mc, mea = np.full(n, 5000), np.full(n, 5000 + 2000 * 10), np.full(n, 10)
    abs_m, chi = [], []
    for seed in range(1, 9):
        o = observables(anneal(4, temps, eq, mc, mea, seed=seed), temps, 4)
        abs_m.append(o["Abs_m"])
        chi.append(o["Susceptibility"])
    abs_m, chi = np.mean(abs_m, axis=0), np.mean(chi, axis=0)
    for i, T in enumerate(temps):
        assert abs(abs_m[i] - exact.moments(T)[0]) < 0.03, T
        assert abs(chi[i] - exact.susceptibility(T)) < 0.15 * max(exact.susceptibility(T), 0.5), T
