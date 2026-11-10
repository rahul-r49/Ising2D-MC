"""Exact results for the 4 x 4 Ising lattice (periodic), by summing over all 2^16 states."""
import numpy as np

L = 4
N = L * L


def _states():
    bits = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) * 2 - 1
    return bits.reshape(-1, L, L)


_S = _states()
_E = -(_S * (np.roll(_S, 1, axis=1) + np.roll(_S, 1, axis=2))).sum(axis=(1, 2))   # each bond once
_M = _S.sum(axis=(1, 2)) / N


def moments(T):
    """Exact <|m|>, <m^2>, <m^4> of the 4 x 4 lattice at temperature T (J = k_B = 1)."""
    w = np.exp(-(_E - _E.min()) / T)
    w /= w.sum()
    return (w * np.abs(_M)).sum(), (w * _M ** 2).sum(), (w * _M ** 4).sum()


def susceptibility(T):
    a, m2, _ = moments(T)
    return N * (m2 - a ** 2) / T
