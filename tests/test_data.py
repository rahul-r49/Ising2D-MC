"""Analysing the stored dissertation data must keep giving the published results.

This guards the analysis code: if a change to src/ising/analysis.py shifts T_c, gamma/nu
or alpha, these tests fail."""
from pathlib import Path

import numpy as np
import pytest

from ising import analysis as A

DATA = Path(__file__).resolve().parents[1] / "data" / "dissertation"


@pytest.fixture(scope="module")
def s():
    return A.summarise(DATA)


def test_lattice_sizes(s):
    assert s["Ls"] == [16, 32, 40, 64, 80, 128]


def test_critical_temperature_from_susceptibility(s):
    assert s["Tc_chi"] == pytest.approx(2.264, abs=0.001)
    assert abs(s["Tc_chi"] - A.TC_EXACT) < 0.01


def test_critical_temperature_from_m0(s):
    assert s["Tc_m0"] == pytest.approx(2.264, abs=0.002)



def test_gamma_over_nu(s):
    assert s["gamma_over_nu"] == pytest.approx(1.751, abs=0.002)
    assert abs(s["gamma_over_nu"] - 1.75) < 0.01


def test_growth_exponent(s):
    assert s["growth_L"] == 512
    assert s["alpha"] == pytest.approx(0.522, abs=0.001)


def test_binder_cumulant_limits():
    for L in (16, 128):
        T, U = A.load_observable(DATA, "U", L)
        assert U[T < 1.0].mean() == pytest.approx(2 / 3, abs=0.005)      # ordered phase
        assert abs(U[T > 4.0].mean()) < 0.05                              # disordered phase
