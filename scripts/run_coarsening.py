"""Domain growth after a quench: average domain size l(t) over many random starts.

    python scripts/run_coarsening.py --L 128 --seeds 200 --jobs 4 --out runs/coarsening_L128

Every seed is an independent run (random initial lattice, quench to temperature T),
so the seeds are spread over --jobs worker processes. Use at most the number of
PHYSICAL cores. The cost grows as  seeds x steps x L^2,  so L = 512 with 200 seeds
takes a very long time; try a small L and few seeds first.

Writes, to the output folder,
    R_<L>_<seed>.csv   l(t) of each seed
    R_<L>_mean.csv     the average over the seeds (one value per measurement time)
The measurement times are ising.coarsening.measurement_times(steps).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ising.coarsening import init_lattice, domain_growth, measurement_times


def one_seed(args):
    L, seed, T, steps = args
    times = measurement_times(steps)
    state = init_lattice(L, seed)
    return seed, domain_growth(L, T, state, steps, times)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--L", type=int, required=True, help="lattice size")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds (default 8)")
    ap.add_argument("--first-seed", type=int, default=1, help="first seed number (default 1)")
    ap.add_argument("--T", type=float, default=0.5, help="quench temperature in J/k_B (default 0.5)")
    ap.add_argument("--steps", type=int, default=20000, help="sweeps per run (default 20000)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.L, k, a.T, a.steps) for k in range(a.first_seed, a.first_seed + a.seeds)]

    results = {}
    with Pool(a.jobs) as pool:
        for seed, R in pool.imap_unordered(one_seed, jobs):
            pd.DataFrame({f"R_{a.L}_{seed}": R}).to_csv(out / f"R_{a.L}_{seed}.csv", index=False)
            results[seed] = R
            print(f"seed {seed} done", flush=True)

    mean = np.mean([results[k] for k in sorted(results)], axis=0)
    pd.DataFrame({"R_mean": mean}).to_csv(out / f"R_{a.L}_mean.csv", index=False)
    print(f"wrote {out / f'R_{a.L}_mean.csv'} (average of {len(results)} seeds)")


if __name__ == "__main__":
    main()
