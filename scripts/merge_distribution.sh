#!/usr/bin/env bash
# Merge the per-seed distribution files into one CSV.
#
#   bash scripts/merge_distribution.sh runs/L64 64
#
# The first argument is the folder with the C++ output, the second the lattice size.
# Result: <folder>/merged_<L>.csv. It has the temperature header once, followed by
# the rows of every seed. With many seeds this file is very large (tens of GB for
# 2000 seeds), so merge only as many seeds as you need.
set -euo pipefail

DIR="${1:?usage: merge_distribution.sh <folder> <L>}"
L="${2:?usage: merge_distribution.sh <folder> <L>}"
OUT="$DIR/merged_${L}.csv"

FILES=( "$DIR"/Run_no_"${L}"_*_Dist_Curve.csv )
[ -e "${FILES[0]}" ] || { echo "no Run_no_${L}_*_Dist_Curve.csv files in $DIR" >&2; exit 1; }

head -1 "${FILES[0]}" > "$OUT"                  # header from the first file
tail -n +2 -q "${FILES[@]}" >> "$OUT"            # data rows (header skipped) from all files
echo "merged ${#FILES[@]} files into $OUT"
