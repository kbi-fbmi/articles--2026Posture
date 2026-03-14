#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Propagate sex (pohlavi) to all rows per patient")
parser.add_argument("input_csv", help="Input CSV file")
parser.add_argument("--output_csv", default="sex_propagated.csv",
                    help="Output CSV file")
args = parser.parse_args()

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(args.input_csv)

# ----------------------------
# Normalize pohlavi
# ----------------------------
df["pohlavi"] = (
    df["pohlavi"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Treat empty strings and 'NAN' as missing
df.loc[df["pohlavi"].isin(["", "NAN", "NONE"]), "pohlavi"] = np.nan

# ----------------------------
# Extract first non-missing sex per patient
# ----------------------------
def first_valid_sex(series):
    for val in series:
        if val in ["M", "F"]:
            return val
    return np.nan

sex_map = (
    df.groupby("study_id")["pohlavi"]
    .apply(first_valid_sex)
)

# ----------------------------
# Propagate to all rows
# ----------------------------
df["pohlavi"] = df["study_id"].map(sex_map)

# ----------------------------
# Save output
# ----------------------------
df.to_csv(args.output_csv, index=False)

print("Sex propagation completed.")
print("Sex distribution:")
print(df["pohlavi"].value_counts(dropna=False))
print(f"\nOutput saved to: {args.output_csv}")
