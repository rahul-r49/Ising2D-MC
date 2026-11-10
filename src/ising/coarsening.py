"""Phase-ordering kinetics: quench from a random state to a low temperature and
follow the average domain size l(t).

Time t is measured in Monte Carlo sweeps (1 sweep = L*L single-spin flip attempts).
This is the Python/Numba code behind the domain-growth results, with the
quench temperature T = 0.5 J/k_B.
"""
import numpy as np
import numba


def measurement_times(steps=20000, n=500):
    """Sweep numbers at which the domain length is measured: logarithmically spaced
    between 1 and `steps`, so there are equally many points per decade.
    (With the defaults this gives 351 distinct times.)"""
    return np.unique(np.geomspace(1, steps, n).astype(int))


@numba.njit(cache=True)
def init_lattice(L, seed):
    """L x L array of random +1/-1 spins, seeded with `seed`.
    The same Numba random stream then continues into domain_growth()."""
    np.random.seed(seed)
    state = np.empty((L, L), dtype=np.int64)
    for x in range(L):
        for y in range(L):
            state[x, y] = 2 * np.random.randint(0, 2) - 1
    return state


@numba.njit(cache=True)
def domain_length(state):
    """Average domain size  l = 2 N / (number of walls in x + number of walls in y).

    A wall is a pair of neighbouring spins that differ. A row (or column) without
    any wall is counted as having one, so that a completely ordered lattice gives
    l = L instead of dividing by zero.
    """
    L = state.shape[0]
    N = L * L
    h_walls = 0
    v_walls = 0
    for x in range(L):
        row = 0
        for y in range(L):
            if state[x, y] != state[x, (y + 1) % L]:
                row += 1
        h_walls += row if row > 0 else 1
    for y in range(L):
        col = 0
        for x in range(L):
            if state[x, y] != state[(x + 1) % L, y]:
                col += 1
        v_walls += col if col > 0 else 1
    return 2.0 * N / (h_walls + v_walls)


@numba.njit(cache=True)
def _sweep(state, L, exp4, exp8):
    """One Metropolis sweep at inverse temperature beta (given as the two Boltzmann factors)."""
    for _ in range(L * L):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        nn = (state[x, (y + 1) % L] + state[x, (y - 1) % L]
              + state[(x - 1) % L, y] + state[(x + 1) % L, y])
        dE = 2 * nn * state[x, y]
        # Metropolis rule. A random number is drawn whenever dE >= 0 (for dE = 0
        # the acceptance probability is 1, so the draw never rejects); this keeps
        # the random stream identical to the dissertation runs.
        if dE < 0:
            state[x, y] = -state[x, y]
        else:
            if dE == 0:
                p = 1.0
            elif dE == 4:
                p = exp4
            else:
                p = exp8
            if np.random.random() < p:
                state[x, y] = -state[x, y]


@numba.njit(cache=True)
def domain_growth(L, T, state, steps, times):
    """Quench: run `steps` sweeps at temperature T, starting from `state`
    (modified in place), and return l(t) at the sweep numbers in `times`.
    Sweeps are numbered from 0, and l is measured right after sweep number t.

    Parameters to change: T (quench temperature), steps (total sweeps) and the
    measurement `times` (see measurement_times()). `steps` must be >= times[-1].
    """
    exp4 = np.exp(-4.0 / T)
    exp8 = np.exp(-8.0 / T)
    is_measured = np.zeros(steps + 1, dtype=np.bool_)
    for t in times:
        is_measured[t] = True

    R = np.empty(len(times))
    k = 0
    for sweep in range(steps + 1):
        _sweep(state, L, exp4, exp8)
        if is_measured[sweep]:
            R[k] = domain_length(state)
            k += 1
    return R


@numba.njit(cache=True)
def snapshots(L, T, state, steps, save_at):
    """Run `steps` sweeps at temperature T and return copies of the lattice after
    the sweep numbers listed in save_at (sweep 0 means the state after one sweep)."""
    exp4 = np.exp(-4.0 / T)
    exp8 = np.exp(-8.0 / T)
    out = []
    for sweep in range(steps + 1):
        _sweep(state, L, exp4, exp8)
        for s in save_at:
            if s == sweep:
                out.append(state.copy())
    return out
