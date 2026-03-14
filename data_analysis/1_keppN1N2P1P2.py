#!/usr/bin/env python3

import pandas as pd
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows where w_ex_test_sel is 2.0, 3.0, 6.0, or 7.0"
    )
    parser.add_argument(
        "RBD_posture_analysis_filtered_csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "N1N2P1P2RBD_posture_analysis_filtered_csv",
        help="Path to output filtered CSV file"
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.RBD_posture_analysis_filtered_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    if "w_ex_test_sel" not in df.columns:
        print("Error: column 'w_ex_test_sel' not found in the CSV file.")
        sys.exit(1)

    # Values to remove
    values_to_remove = [2.0, 3.0, 6.0, 7.0]

    # Keep rows that are NOT in values_to_remove
    # NaN values are preserved automatically
    df_filtered = df[~df["w_ex_test_sel"].isin(values_to_remove)]

    try:
        df_filtered.to_csv(args.N1N2P1P2RBD_posture_analysis_filtered_csv, index=False)
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        sys.exit(1)

    print("Filtering complete.")
    print(f"Input rows   : {len(df)}")
    print(f"Kept rows    : {len(df_filtered)}")
    print(f"Removed rows : {len(df) - len(df_filtered)}")


if __name__ == "__main__":
    main()
