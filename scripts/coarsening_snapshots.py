"""Snapshots of the lattice at several times after a quench (white = -1, blue = +1).

    python scripts/coarsening_snapshots.py --L 256 --times 0 10 100 1000

The figure is saved in results/ with a name made from the options, or at --out.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ising.coarsening import init_lattice, snapshots


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--L", type=int, default=256, help="lattice size (default 256)")
    ap.add_argument("--times", type=int, nargs="+", default=[0, 10, 100, 1000],
                    help="sweep numbers to show (default 0 10 100 1000)")
    ap.add_argument("--T", type=float, default=0.5, help="quench temperature (default 0.5)")
    ap.add_argument("--seed", type=int, default=3, help="random seed (default 3)")
    ap.add_argument("--out", help="output image (default: results/<name>.png)")
    a = ap.parse_args()

    times = sorted(a.times)
    state = init_lattice(a.L, a.seed)
    snaps = snapshots(a.L, a.T, state, times[-1], np.array(times))

    n = len(times)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False)
    cmap = mcolors.ListedColormap(["white", "blue"])
    for ax, t, snap in zip(axes.ravel(), times, snaps):
        ax.imshow(snap, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
        ax.set_xlabel(f"t = {t} MCS")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()

    name = f"coarsening_snapshots_L{a.L}_T{a.T:g}_seed{a.seed}"
    out = Path(a.out) if a.out else Path("results") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
