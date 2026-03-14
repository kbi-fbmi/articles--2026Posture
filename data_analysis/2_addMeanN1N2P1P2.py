#!/usr/bin/env python3

import pandas as pd
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Add N1N2 and P1P2 gait metrics to followup summary rows"
    )
    parser.add_argument("input_csv", help="Input CSV file")

    args = parser.parse_args()
    output_csv = "GAITmeanN1N2P1P2.csv"

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    required_cols = [
        "study_id",
        "redcap_event_name",
        "redcap_repeat_instrument",
        "redcap_repeat_instance",
        "w_ex_gr_cad_r",
        "w_ex_gr_velo_l",
        "w_ex_time",
        "w_ex_gr_stc_l_t",
        "w_ex_gr_stc_r_t",
        "w_ex_test_t___0",
        "w_ex_test_t___1",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        sys.exit(1)

    # New columns
    new_cols = [
        "w_ex_gr_cad_r_meanN1N2",
        "w_ex_gr_cad_r_meanP1P2",
        "w_ex_gr_velo_l_meanN1N2",
        "w_ex_gr_velo_l_meanP1P2",
        "w_ex_time_meanN1N2",
        "w_ex_time_meanP1P2",
        "w_ex_gr_stc_l_t_meanN1N2",
        "w_ex_gr_stc_l_t_meanP1P2",
        "w_ex_gr_stc_r_t_meanN1N2",
        "w_ex_gr_stc_r_t_meanP1P2",
        "w_ex_test_t___0_N1N2",
        "w_ex_test_t___0_P1P2",
        "w_ex_test_t___1_N1N2",
        "w_ex_test_t___1_P1P2",
    ]

    for col in new_cols:
        df[col] = pd.NA

    df["redcap_repeat_instance"] = pd.to_numeric(
        df["redcap_repeat_instance"], errors="coerce"
    )

    # Process per patient
    for study_id, g in df.groupby("study_id"):

        followup_summary_mask = (
            g["redcap_event_name"].str.startswith("followup", na=False)
            & g["redcap_repeat_instrument"].isna()
        )

        if followup_summary_mask.sum() != 1:
            continue

        idx = g[followup_summary_mask].index[0]

        exam_rows = g[
            g["redcap_event_name"].str.startswith("followup", na=False)
            & g["redcap_repeat_instrument"].notna()
        ]

        n1n2 = exam_rows[exam_rows["redcap_repeat_instance"].isin([1.0, 2.0])]
        p1p2 = exam_rows[exam_rows["redcap_repeat_instance"].isin([5.0, 6.0])]

        # Means
        df.loc[idx, "w_ex_gr_cad_r_meanN1N2"] = n1n2["w_ex_gr_cad_r"].mean()
        df.loc[idx, "w_ex_gr_cad_r_meanP1P2"] = p1p2["w_ex_gr_cad_r"].mean()
        df.loc[idx, "w_ex_gr_velo_l_meanN1N2"] = n1n2["w_ex_gr_velo_l"].mean()
        df.loc[idx, "w_ex_gr_velo_l_meanP1P2"] = p1p2["w_ex_gr_velo_l"].mean()
        df.loc[idx, "w_ex_time_meanN1N2"] = n1n2["w_ex_time"].mean()
        df.loc[idx, "w_ex_time_meanP1P2"] = p1p2["w_ex_time"].mean()
        df.loc[idx, "w_ex_gr_stc_l_t_meanN1N2"] = n1n2["w_ex_gr_stc_l_t"].mean()
        df.loc[idx, "w_ex_gr_stc_l_t_meanP1P2"] = p1p2["w_ex_gr_stc_l_t"].mean()
        df.loc[idx, "w_ex_gr_stc_r_t_meanN1N2"] = n1n2["w_ex_gr_stc_r_t"].mean()
        df.loc[idx, "w_ex_gr_stc_r_t_meanP1P2"] = p1p2["w_ex_gr_stc_r_t"].mean()

        # Single-instance extraction (1 and 5)
        row_1 = exam_rows[exam_rows["redcap_repeat_instance"] == 1.0]
        row_5 = exam_rows[exam_rows["redcap_repeat_instance"] == 5.0]

        if not row_1.empty:
            df.loc[idx, "w_ex_test_t___0_N1N2"] = row_1["w_ex_test_t___0"].iloc[0]
            df.loc[idx, "w_ex_test_t___1_N1N2"] = row_1["w_ex_test_t___1"].iloc[0]

        if not row_5.empty:
            df.loc[idx, "w_ex_test_t___0_P1P2"] = row_5["w_ex_test_t___0"].iloc[0]
            df.loc[idx, "w_ex_test_t___1_P1P2"] = row_5["w_ex_test_t___1"].iloc[0]

    # =====================
    # FINAL CLEANUP
    # =====================

    df = df[df["redcap_repeat_instrument"] != "walk_exam"]

    df = df.drop(
        columns=[
            "pd_duration",
            "redcap_repeat_instance",
            "redcap_repeat_instrument",
            "w_ex_test_sel",
            "w_ex_test_t___0",
            "w_ex_test_t___1",
            "w_ex_gr_cad_r",
            "w_ex_gr_velo_l",
            "w_ex_gr_stc_l_t",
            "w_ex_gr_stc_r_t",
            "w_ex_time",
        ],
        errors="ignore",
    )

    df.to_csv(output_csv, index=False)
    print(f"Output written to: {output_csv}")


if __name__ == "__main__":
    main()
