"""Tests of the C++ program (cpp/ising.cpp), built with the very short QUICK schedule."""
import numpy as np
import pandas as pd
import pytest

import exact
from ising import temperatures


def column(path):
    return pd.read_csv(path).iloc[:, 0].to_numpy()



def test_observables_agree_with_recorded_magnetisations(build, run, tmp_path):
    """m, |m|, chi and U4 written by the program must follow from the raw m values."""
    L, N = 16, 16 * 16
    run(build(L, 1), tmp_path)
    dist = pd.read_csv(tmp_path / "Run_no_16_1_Dist_Curve.csv")
    T = dist.columns.astype(float).to_numpy()
    assert np.allclose(T, temperatures.temperature_grid())
    m = dist.to_numpy()
    abs_m, m2, m4 = np.abs(m).mean(0), (m ** 2).mean(0), (m ** 4).mean(0)
    np.testing.assert_allclose(column(tmp_path / "m_16_1.csv"), m.mean(0), atol=1e-12)
    np.testing.assert_allclose(column(tmp_path / "Abs_m_16_1.csv"), abs_m, rtol=1e-12)
    np.testing.assert_allclose(column(tmp_path / "Sus_16_1.csv"), N * (m2 - abs_m ** 2) / T, rtol=1e-9)
    np.testing.assert_allclose(column(tmp_path / "U_16_1.csv"), 1 - m4 / (3 * m2 ** 2), rtol=1e-9)


def test_magnetisation_values_are_allowed(build, run, tmp_path):
    """On an L x L lattice m is a multiple of 2/L^2 between -1 and 1."""
    run(build(16, 1), tmp_path)
    m = pd.read_csv(tmp_path / "Run_no_16_1_Dist_Curve.csv").to_numpy()
    assert np.abs(m).max() <= 1
    steps = (m * 256 + 256) / 2                     # (M + N) / 2 is an integer
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)


def test_high_temperature_is_disordered(build, run, tmp_path):
    run(build(16, 2), tmp_path)
    for k in (1, 2):
        assert column(tmp_path / f"Abs_m_16_{k}.csv")[0] < 0.15          # T = 4.5
        assert abs(column(tmp_path / f"U_16_{k}.csv")[0]) < 0.1


def test_same_seed_gives_identical_files(build, run, tmp_path):
    exe = build(16, 1)
    run(exe, tmp_path / "a")
    run(exe, tmp_path / "b")
    for p in (tmp_path / "a").iterdir():
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes()


def test_result_does_not_depend_on_thread_count(build, run, tmp_path):
    exe = build(16, 4)
    run(exe, tmp_path / "t1", threads=1)
    run(exe, tmp_path / "t2", threads=2)
    for p in (tmp_path / "t1").iterdir():
        assert p.read_bytes() == (tmp_path / "t2" / p.name).read_bytes()



def test_matches_exact_4x4_lattice(build, run, tmp_path):
    """Averaged over seeds, |m| and chi of a 4 x 4 lattice must agree with the exact
    enumeration of all 2^16 states."""
    n_seeds = 16
    run(build(4, n_seeds), tmp_path)
    ks = range(1, n_seeds + 1)
    abs_m = np.mean([column(tmp_path / f"Abs_m_4_{k}.csv") for k in ks], axis=0)
    chi = np.mean([column(tmp_path / f"Sus_4_{k}.csv") for k in ks], axis=0)
    T = temperatures.temperature_grid()
    for i in range(0, len(T), 6):
        a, _, _ = exact.moments(T[i])
        assert abs(abs_m[i] - a) < 0.01, (T[i], abs_m[i], a)
        assert abs(chi[i] - exact.susceptibility(T[i])) < 0.06 * max(exact.susceptibility(T[i]), 0.5), T[i]


def test_matches_exact_3x3_lattice(build, run, tmp_path):
    """L need not be a power of 2: a 3 x 3 lattice (2^9 states) must also agree with exact enumeration."""
    L, N = 3, 9
    bits = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) * 2 - 1
    S = bits.reshape(-1, L, L)
    E = -(S * (np.roll(S, 1, axis=1) + np.roll(S, 1, axis=2))).sum(axis=(1, 2))
    M = np.abs(S.sum(axis=(1, 2))) / N
    n_seeds = 16
    run(build(L, n_seeds), tmp_path)
    abs_m = np.mean([column(tmp_path / f"Abs_m_3_{k}.csv") for k in range(1, n_seeds + 1)], axis=0)
    T = temperatures.temperature_grid()
    for i in range(0, len(T), 6):
        w = np.exp(-(E - E.min()) / T[i])
        w /= w.sum()
        assert abs(abs_m[i] - (w * M).sum()) < 0.01, (T[i], abs_m[i], (w * M).sum())



def test_both_sources_give_identical_output_for_power_of_two(build, run, tmp_path):
    """ising.cpp (bit shift) and ising_any_L.cpp (multiply-shift) pick the same sites when
    L is a power of 2, so every output file must be identical."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    run(build(16, 1, source="ising.cpp"), a)
    run(build(16, 1, source="ising_any_L.cpp"), b)
    names = sorted(p.name for p in a.iterdir())
    assert len(names) == 5 and names == sorted(p.name for p in b.iterdir())
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


# (equilibration, measurement interval) for T > 2.5, 2.0 <= T <= 2.5, T < 2.0
PRESETS = {
    16:  [(500_000, 500),   (7_000_000, 25_000),     (300_000, 1_000)],
    32:  [(500_000, 500),   (30_000_000, 25_000),    (300_000, 1_000)],
    40:  [(500_000, 1_000), (50_000_000, 50_000),    (300_000, 1_000)],
    64:  [(500_000, 500),   (50_000_000, 100_000),   (300_000, 1_000)],
    80:  [(3_000_000, 31_000), (200_000_000, 250_000), (3_000_000, 15_000)],
    128: [(3_000_000, 100_000), (300_000_000, 2_000_000), (3_000_000, 50_000)],
}


def expected_attempts(L):
    """Total attempts per seed from the table of presets (5000 measurements per temperature)."""
    T = temperatures.temperature_grid()
    if L in PRESETS:
        table = PRESETS[L]
    else:   # scaled from L = 64: (L/64)^(2+z), z = 2.2 near Tc and 2 elsewhere
        table = []
        for r, z in enumerate((2.0, 2.2, 2.0)):
            f = (L / 64) ** (2 + z)
            eq, mea = PRESETS[64][r]
            table.append((round(eq * f), round(mea * f)))
    total = 0
    for t in T:
        eq, mea = table[0 if t > 2.5 else (1 if t >= 2.0 else 2)]
        total += eq + 5000 * mea
    return total


@pytest.mark.parametrize("L", [16, 32, 40, 64, 80, 128, 100])
def test_step_presets(build, tmp_path, L):
    """The program reports the planned number of attempts per seed; it must match the
    documented settings (5000 measurements per temperature; scaling rule for other L)."""
    import subprocess
    exe = build(L, 1, quick=False)
    p = subprocess.Popen([str(exe), str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        line = next(l for l in iter(p.stdout.readline, "") if l.startswith("Attempts per seed"))
    finally:
        p.kill()
        p.wait()
    reported = int(line.split(":")[1].split()[0])
    assert abs(reported - expected_attempts(L)) <= 0.0001 * reported
