"""Convert the merged distribution CSV to Parquet, which loads far faster and lets
the plotting code read a single temperature column without reading the whole file.

    python scripts/convert_to_parquet.py runs/L64/merged_64.csv

Writes runs/L64/merged_64.parquet (or the path given with --out). The file is read
in chunks, so it does not need to fit in memory. Column names are the temperatures
with the decimal point replaced by an underscore: T = 2.50 becomes '2_50'.
Needs pyarrow:  pip install -e ".[dist]"
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="merged CSV file made by merge_distribution.sh")
    ap.add_argument("--out", help="output .parquet file (default: next to the CSV)")
    ap.add_argument("--chunk", type=int, default=5000, help="rows read at a time (lower it if memory is short)")
    a = ap.parse_args()

    src = Path(a.csv)
    dst = Path(a.out) if a.out else src.with_suffix(".parquet")

    raw = pd.read_csv(src, nrows=0).columns.tolist()
    clean = [f"{round(float(c), 2):.2f}".replace(".", "_") for c in raw]
    schema = pa.schema([pa.field(c, pa.float64()) for c in clean])

    writer = pq.ParquetWriter(dst, schema)
    for chunk in pd.read_csv(src, chunksize=a.chunk, dtype={c: "float64" for c in raw}):
        chunk.columns = clean
        writer.write_table(pa.Table.from_pandas(chunk, schema=schema))
        del chunk
        gc.collect()
    writer.close()
    print("wrote", dst)


if __name__ == "__main__":
    main()
