"""Python (Numba) reference implementation of the equilibrium simulation.

This is the same algorithm as cpp/ising.cpp, written to be easy to read. It was
the starting point for the C++ code. It uses a different random number generator,
so a given seed gives different numbers than the C++ program, but both sample the
same distribution. It is much slower than the C++ code, so use it for small
lattices and short runs, and use the C++ program for the real data.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@numba.njit(cache=True)
def _random_lattice(L):
    """L x L array of random +1/-1 spins (uses Numba's own random generator)."""
    state = np.empty((L, L), dtype=np.int64)
    for x in range(L):
        for y in range(L):
            state[x, y] = 2 * np.random.randint(0, 2) - 1
    return state


@numba.njit(cache=True)
def _anneal(L, T, state, mc_steps, eq_steps, mea_steps):
    """Metropolis annealing through the temperatures in T (highest first).

    One "step" is one single-spin flip attempt (L*L steps make one sweep).
    Returns a list with, for every temperature, an array of the recorded
    magnetisations per spin.
    """
    # Periodic neighbours. int64 is needed: int8 tables overflow for L > 128.
    right = np.arange(1, L + 1)
    right[-1] = 0
    left = np.arange(-1, L - 1)
    left[0] = L - 1

    out = []
    for t in range(len(T)):
        n_meas = (mc_steps[t] - 1) // mea_steps[t] - (eq_steps[t] + mea_steps[t] - 1) // mea_steps[t] + 1
        mags = np.zeros(n_meas)
        meas = 0
        M = np.sum(state)                      # total magnetisation, updated incrementally
        beta = 1.0 / T[t]
        exp4 = np.exp(-4.0 * beta)             # the only two Boltzmann factors needed
        exp8 = np.exp(-8.0 * beta)

        for i in range(mc_steps[t]):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            nn = (state[x, right[y]] + state[x, left[y]]
                  + state[left[x], y] + state[right[x], y])
            dE = 2 * nn * state[x, y]          # energy change if this spin flips (J = 1)

            # Metropolis rule: always accept dE < 0; otherwise accept with
            # probability exp(-dE/T). (A random number is drawn for dE = 0 as
            # well; it never rejects, because the probability is 1.)
            accept = False
            if dE < 0:
                accept = True
            else:
                if dE == 0:
                    p = 1.0
                elif dE == 4:
                    p = exp4
                else:
                    p = exp8
                accept = np.random.random() < p

            if accept:
                state[x, y] = -state[x, y]
                M += 2 * state[x, y]

            if i >= eq_steps[t] and i % mea_steps[t] == 0:
                mags[meas] = M / (L * L)
                meas += 1
        out.append(mags)
    return out


def anneal(L, temps, eq_steps, mc_steps, mea_steps, seed=1):
    """Run one seed. `temps`, `eq_steps`, `mc_steps` and `mea_steps` are arrays with
    one entry per temperature (highest T first). Returns a list of 1D arrays of
    recorded magnetisations per spin, one array per temperature.

    Parameters to change: the temperatures, and the number of steps per temperature.
    The step counts are single-spin attempts and must satisfy eq < mc.
    """
    temps = np.asarray(temps, dtype=np.float64)
    eq_steps = np.asarray(eq_steps, dtype=np.int64)
    mc_steps = np.asarray(mc_steps, dtype=np.int64)
    mea_steps = np.asarray(mea_steps, dtype=np.int64)
    _seed(seed)
    state = _random_lattice(L)
    return _anneal(L, temps, state, mc_steps, eq_steps, mea_steps)


def observables(mags, temps, L):
    """<m>, <|m|>, susceptibility and Binder cumulant from the recorded magnetisations.

        chi = L^2 (<m^2> - <|m|>^2) / T          (k_B = 1)
        U4  = 1 - <m^4> / (3 <m^2>^2)
    """
    N = L * L
    m = np.array([np.mean(a) for a in mags])
    abs_m = np.array([np.mean(np.abs(a)) for a in mags])
    m2 = np.array([np.mean(a ** 2) for a in mags])
    m4 = np.array([np.mean(a ** 4) for a in mags])
    chi = N * (m2 - abs_m ** 2) / np.asarray(temps)
    U4 = 1.0 - m4 / (3.0 * m2 ** 2)
    return {"m": m, "Abs_m": abs_m, "Susceptibility": chi, "U": U4}
