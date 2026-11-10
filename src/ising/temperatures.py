"""Temperature grids (in units of J / k_B), listed from the highest to the lowest T,
which is the order in which the annealing visits them."""
import numpy as np


def _descending(start, stop, step):
    """Values start, start-step, ... that stay above `stop` (rounded to 2 decimals).
    Same rule as the C++ code, which avoids floating-point drift."""
    out, t = [], start
    while round(t, 2) > stop:
        out.append(round(t, 2))
        t -= step
    return out


def temperature_grid():
    """The 131-point schedule used by cpp/ising.cpp and by the dissertation results
    for L = 32, 40, 64, 80 and 128.

        4.5  ... 2.6   step 0.10   (20 points)
        2.5  ... 1.46  step 0.01   (105 points, dense around Tc = 2.269)
        1.25 ... 0.5   step 0.15   (6 points)
    """
    return np.array(_descending(4.5, 2.5, 0.10)
                    + _descending(2.5, 1.45, 0.01)
                    + _descending(1.25, 0.45, 0.15))


def legacy_grid_L16():
    """The 150-point grid of the L = 16 results in data/dissertation.

    It was built with np.arange in the dissertation notebooks: 4.5 ... 2.8 in
    steps of 0.10 (18 points), 2.7 ... 1.45 in steps of 0.01 (126 points) and
    1.25 ... 0.5 in steps of 0.15 (6 points). It is kept exactly as it was so
    that the stored L = 16 columns line up with their temperatures."""
    return np.concatenate((np.arange(4.5, 2.7, -0.10),
                           np.arange(2.7, 1.45, -0.01),
                           np.arange(1.25, 0.45, -0.15)))


def grid_matching(n_points):
    """The temperature grid that has `n_points` entries: the standard 131-point grid
    (output of cpp/ising.cpp) or the 150-point legacy L = 16 grid."""
    for grid in (temperature_grid(), legacy_grid_L16()):
        if len(grid) == n_points:
            return grid
    raise ValueError(f"no known temperature grid has {n_points} points "
                     f"(expected {len(temperature_grid())} or {len(legacy_grid_L16())})")
