import numpy as np

from ising.coarsening import domain_length


def test_domain_length_of_ordered_lattice_is_L():
    assert domain_length(np.ones((16, 16), dtype=np.int64)) == 16


def test_domain_length_of_checkerboard_is_1():
    x, y = np.indices((16, 16))
    assert domain_length(((x + y) % 2 * 2 - 1).astype(np.int64)) == 1


def test_domain_length_of_stripes():
    state = np.ones((16, 16), dtype=np.int64)
    state[:, 8:] = -1                        # two vertical stripes of width 8
    # every row has 2 walls (32 in total); every column has none and counts as 1 (16)
    assert np.isclose(domain_length(state), 2 * 256 / (32 + 16))
