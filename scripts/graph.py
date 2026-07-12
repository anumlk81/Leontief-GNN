"""
build_A_matrix.py
Converts cleaned $ flow matrices into technical coefficient (A) matrices,
for every year found in DATA/clean/.

A[i,j] = flow[i,j] / total_output[j]
  -> dollars of input from industry i needed per $1 of industry j's output

Run: python scripts/build_A_matrix.py
Reads:  DATA/clean/clean_flows_<year>.csv, DATA/clean/clean_output_<year>.csv
Writes: DATA/matrices/A_<year>.csv
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

CLEAN_DIR = Path(r"C:\Users\Lenovo\Desktop\projects\Leontief-GNN\datasets\clean_datasets")
OUT_DIR = Path(r"C:\Users\Lenovo\Desktop\projects\Leontief-GNN\datasets\matrices")


def find_years(clean_dir: Path):
    years = []
    for f in clean_dir.glob("clean_flows_*.csv"):
        match = re.search(r"clean_flows_(\d{4})\.csv", f.name)
        if match:
            years.append(match.group(1))
    return sorted(years)


def build_A_matrix(flows: pd.DataFrame, total_output: pd.Series) -> pd.DataFrame:
    denom = total_output.replace(0, np.nan)
    A = flows.div(denom, axis=1).fillna(0.0)
    return A


def validate_A_matrix(A: pd.DataFrame, year: str):
    n = A.shape[0]
    I = np.eye(n)

    try:
        np.linalg.inv(I - A.values)
    except np.linalg.LinAlgError:
        print(f"[{year}] WARNING: (I - A) is singular — Leontief inverse does not exist.")
        return

    col_sums = A.sum(axis=0)
    print(f"[{year}] {n} industries | column-sum range: "
          f"{col_sums.min():.3f} to {col_sums.max():.3f}")

    bad = col_sums[col_sums >= 1.0]
    if len(bad) > 0:
        print(f"[{year}] WARNING: {len(bad)} industries have column sum >= 1.0: "
              f"{list(bad.index)}")
    else:
        print(f"[{year}] OK — Leontief inverse computed, all column sums < 1.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = find_years(CLEAN_DIR)

    if not years:
        print(f"No clean_flows_*.csv files found in {CLEAN_DIR}/")
        return

    print(f"Found years: {years}\n")

    for year in years:
        flows_path = CLEAN_DIR / f"clean_flows_{year}.csv"
        output_path = CLEAN_DIR / f"clean_output_{year}.csv"

        if not output_path.exists():
            print(f"[{year}] SKIPPED — missing {output_path.name}")
            continue

        flows = pd.read_csv(flows_path, index_col=0)
        total_output = pd.read_csv(output_path, index_col=0).iloc[:, 0]

        A = build_A_matrix(flows, total_output)
        validate_A_matrix(A, year)

        A.to_csv(OUT_DIR / f"A_{year}.csv")
        print(f"[{year}] Saved A_{year}.csv -> shape {A.shape}\n")


if __name__ == "__main__":
    main()