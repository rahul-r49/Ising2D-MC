"""Average the per-seed output of the C++ program over all seeds.

    python scripts/average_seeds.py runs/L64 --L 64

reads  m_64_<seed>.csv, Abs_m_64_<seed>.csv, Sus_64_<seed>.csv, U_64_<seed>.csv
from the folder and writes the seed averages  m_64.csv, Abs_m_64.csv, Sus_64.csv,
U_64.csv  (one value per temperature, highest temperature first) to the same
folder, or to the folder given with --out.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# file prefix -> column header of the averaged file (the headers used in data/dissertation)
OBSERVABLES = {"m": "m", "Abs_m": "Abs_m", "Sus": "Susceptibility", "U": "U"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="folder with the per-seed CSV files")
    ap.add_argument("--L", type=int, required=True, help="lattice size of the runs")
    ap.add_argument("--out", help="folder for the averaged files (default: same folder)")
    a = ap.parse_args()

    folder = Path(a.folder)
    out = Path(a.out) if a.out else folder
    out.mkdir(parents=True, exist_ok=True)

    seeds = sorted(int(m.group(1)) for p in folder.glob(f"U_{a.L}_*.csv")
                   if (m := re.fullmatch(rf"U_{a.L}_(\d+)\.csv", p.name)))
    if not seeds:
        raise SystemExit(f"no U_{a.L}_<seed>.csv files found in {folder}")

    for prefix, header in OBSERVABLES.items():
        columns = [pd.read_csv(folder / f"{prefix}_{a.L}_{k}.csv").iloc[:, 0].to_numpy()
                   for k in seeds]
        mean = np.mean(columns, axis=0)
        pd.DataFrame({header: mean}).to_csv(out / f"{prefix}_{a.L}.csv", index=False)
    print(f"averaged {len(seeds)} seeds (L = {a.L}); wrote 4 files to {out}")


if __name__ == "__main__":
    main()
