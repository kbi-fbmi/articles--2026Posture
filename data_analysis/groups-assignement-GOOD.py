#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Assign clinical groups (RBD, CON, TD, PIGD)")
parser.add_argument("input_csv", help="Input CSV file")
parser.add_argument("--output_csv", default="grouped_merged-dates_gait-meanN1N2_output.csv",
                    help="Output CSV file with group column")
args = parser.parse_args()

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(args.input_csv)

# ----------------------------
# Group assignment function
# ----------------------------
def assign_group(row):
    study_id = str(row["study_id"]).upper()

    # ---- Non-PD groups ----
    if study_id.startswith("RBD"):
        return "RBD"

    if study_id.startswith("CON"):
        return "CON"

    # ---- PD patients ----
    if study_id.startswith("BIO-PD"):
        td = row.get("mdsupdrs_td", np.nan)
        pigd = row.get("mdsupdrs_pigd", np.nan)
        ratio = row.get("mdsupdrs_r", np.nan)

        # Priority rules
        if pd.notna(td) and pd.notna(pigd):
            if td > 0 and pigd == 0:
                return "TD"
            if td == 0 and pigd > 0:
                return "PIGD"

        if pd.notna(ratio):
            if ratio >= 1.15:
                return "TD"
            if ratio <= 0.9:
                return "PIGD"

        return "PD_UNCLASSIFIED"

    # ---- Unknown ----
    return "UNKNOWN"

# ----------------------------
# Apply grouping
# ----------------------------
df["group"] = df.apply(assign_group, axis=1)

# ----------------------------
# Save output
# ----------------------------
df.to_csv(args.output_csv, index=False)

print("Grouping completed.")
print("Group counts:")
print(df["group"].value_counts())
print(f"\nOutput saved to: {args.output_csv}")
