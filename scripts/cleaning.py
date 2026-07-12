"""
clean_use_table.py
Cleans a raw BEA Use Table CSV into a tidy format.

Usage (command line, one file at a time):
    python clean_use_table.py <input_csv> <year> <output_dir>

Usage (script mode, one or many files):
    1. Edit the INPUT_FILES list below to add your CSV paths and years.
    2. Run:  python clean_use_table.py
    (No command-line arguments needed in this mode.)

Outputs (per year):
  <output_dir>/clean_flows_<year>.csv     -> industry x industry $ flow matrix
  <output_dir>/clean_output_<year>.csv    -> total industry output per industry
  <output_dir>/clean_names_<year>.csv     -> industry code -> name mapping
"""
import pandas as pd
import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# SCRIPT-MODE CONFIG
# Add your CSV files here as (path, year) pairs, e.g.:
#   ("data/use_table_2017.csv", "2017"),
#   ("data/use_table_2018.csv", "2018"),
# This is used only if you run the script with NO command-line arguments.
# ---------------------------------------------------------------------------
INPUT_FILES = [
    (r"C:\Users\Lenovo\Desktop\projects\Leontief-GNN\datasets\IOUse_Before_Redefinitions_PUR_Summary-2007.csv", "2007"),
    (r"C:\Users\Lenovo\Desktop\projects\Leontief-GNN\datasets\IOUse_Before_Redefinitions_PUR_Summary-2012.csv", "2012"),
    (r"C:\Users\Lenovo\Desktop\projects\Leontief-GNN\datasets\IOUse_Before_Redefinitions_PUR_Summary-2017.csv", "2017")

    
]

OUTPUT_DIR = "datasets/clean_datasets"


def clean_use_table(csv_path):
    raw = pd.read_csv(csv_path, header=None)

    # locate header rows
    header_row_idx = next(i for i in range(15) if raw.iloc[i, 0] == "IOCode")
    codes_row = raw.iloc[header_row_idx - 1]

    # industry rows span from just after header to the 'Used' (scrap) row
    start = header_row_idx + 1
    end = next(i for i in range(start, len(raw)) if str(raw.iloc[i, 0]).strip() == "Used")

    industry_rows = raw.iloc[start:end].reset_index(drop=True)
    row_codes = industry_rows.iloc[:, 0].astype(str).str.strip().tolist()
    row_names = industry_rows.iloc[:, 1].astype(str).str.strip().tolist()
    row_code_set = set(row_codes)

    # commodity columns (finer-grained than industry rows for retail trade)
    col_end = 2
    while col_end < codes_row.shape[0] and pd.notna(codes_row.iloc[col_end]):
        col_end += 1
    raw_col_codes = codes_row.iloc[2:col_end].astype(str).str.strip().tolist()

    # clean numeric block: strip commas, convert '...' (suppressed/zero) to 0
    raw_block = industry_rows.iloc[:, 2:col_end].copy()
    raw_block = raw_block.replace({r"\.\.\.": "0", ",": ""}, regex=True)
    raw_block = raw_block.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw_block.index = row_codes
    raw_block.columns = raw_col_codes

    # keep only commodity columns with a matching industry row, align order
    flows = raw_block.loc[:, [c for c in raw_col_codes if c in row_code_set]]
    flows = flows.reindex(columns=row_codes, fill_value=0.0)

    # Total Industry Output row (bottom of sheet) — correct denominator, NOT
    # the 'Total Commodity Output' column (different BEA concept)
    tio_row_idx = next(
        i for i in range(end, end + 10)
        if str(raw.iloc[i, 1]).strip() == "Total Industry Output"
    )
    tio = raw.iloc[tio_row_idx, 2:col_end].astype(str).str.replace(",", "", regex=False)
    tio = pd.to_numeric(tio, errors="coerce").fillna(0.0)
    tio.index = raw_col_codes
    total_output = tio.reindex(row_codes, fill_value=0.0)
    total_output.name = "total_output"

    names = pd.Series(row_names, index=row_codes, name="name")

    return flows, total_output, names


def process_one(csv_path, year, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"[skip] Year {year}: file not found -> {csv_path}")
        return

    flows, total_output, names = clean_use_table(csv_path)

    flows.to_csv(os.path.join(out_dir, f"clean_flows_{year}.csv"))
    total_output.to_csv(os.path.join(out_dir, f"clean_output_{year}.csv"))
    names.to_csv(os.path.join(out_dir, f"clean_names_{year}.csv"))

    print(f"Year {year}: cleaned {flows.shape[0]} industries.")
    print(flows.iloc[:3, :3])


if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Command-line mode: python clean_use_table.py <input_csv> <year> <output_dir>
        csv_path, year, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
        process_one(csv_path, year, out_dir)

    elif INPUT_FILES:
        # Script-config mode: files listed in INPUT_FILES above
        for csv_path, year in INPUT_FILES:
            process_one(csv_path, year, OUTPUT_DIR)

    else:
        sys.exit(
            "No input specified.\n"
            "Either run: python clean_use_table.py <input_csv> <year> <output_dir>\n"
            "or add (csv_path, year) tuples to INPUT_FILES at the top of this script "
            "and run it with no arguments."
        )