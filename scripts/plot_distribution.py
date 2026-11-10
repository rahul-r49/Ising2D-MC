"""Magnetisation distribution curves from the merged Parquet file(s).

Counts at several temperatures for one lattice (the distribution splits into two
peaks below the critical temperature and has one peak above it):

    python scripts/plot_distribution.py counts runs/L64/merged_64.parquet --L 64 --temps 2.30 2.34 2.50

Normalised distribution P(m) at one temperature for several lattice sizes
(the peak narrows as the lattice grows):

    python scripts/plot_distribution.py density 16:runs/L16/merged_16.parquet 32:runs/L32/merged_32.parquet --temp 3.0

The figure is saved in results/ with a name made from the options, or at the path
given with --out. Needs pyarrow:  pip install -e ".[dist]"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def column(temp):
    return f"{temp:.2f}".replace(".", "_")


def read(path, temp):
    return pq.read_table(path, columns=[column(temp)]).to_pandas().iloc[:, 0].to_numpy()


def bins_for(L):
    edges = np.linspace(-1, 1, L ** 2 + 1)         # m takes values in steps of 2/L^2
    return edges, (edges[:-1] + edges[1:]) / 2


def counts(a):
    fig, ax = plt.subplots(figsize=(7, 7))
    edges, centres = bins_for(a.L)
    for temp, marker in zip(a.temps, "osv^D"):
        c, _ = np.histogram(read(a.parquet, temp), bins=edges)
        ax.scatter(centres, c, marker=marker, s=7, label=f"T = {temp:.2f}")
    ax.axvline(0, color="blue", lw=0.8)
    ax.set_xlabel("Magnetisation per spin, m")
    ax.set_ylabel("Counts")
    ax.set_title(f"L = {a.L}")
    ax.legend(frameon=False)
    name = f"distribution_counts_L{a.L}_T" + "_".join(f"{t:.2f}" for t in a.temps)
    return fig, name


def density(a):
    fig, ax = plt.subplots(figsize=(7, 7))
    for spec, marker in zip(a.specs, "osv^D"):
        L, path = spec.split(":", 1)
        edges, centres = bins_for(int(L))
        c, _ = np.histogram(read(path, a.temp), bins=edges)
        width = edges[1] - edges[0]
        ax.scatter(centres, c / (c.sum() * width), marker=marker, s=7, label=f"L = {L}")
    ax.axvline(0, color="blue", lw=0.8)
    ax.set_xlabel("Magnetisation per spin, m")
    ax.set_ylabel("P(m)")
    ax.set_title(f"T = {a.temp:.2f}")
    ax.legend(frameon=False)
    name = f"distribution_density_T{a.temp:.2f}_L" + "_".join(s.split(":")[0] for s in a.specs)
    return fig, name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("counts", help="counts at several temperatures, one lattice")
    c.add_argument("parquet")
    c.add_argument("--L", type=int, required=True)
    c.add_argument("--temps", type=float, nargs="+", required=True)
    d = sub.add_parser("density", help="P(m) at one temperature, several lattices")
    d.add_argument("specs", nargs="+", help="L:path pairs, e.g. 16:runs/L16/merged_16.parquet")
    d.add_argument("--temp", type=float, required=True)
    for p in (c, d):
        p.add_argument("--out", help="output image (default: results/<name>.png)")
    a = ap.parse_args()

    fig, name = counts(a) if a.mode == "counts" else density(a)
    out = Path(a.out) if a.out else Path("results") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
