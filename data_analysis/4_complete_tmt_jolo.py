import pandas as pd
import os
import argparse

def complete_tmt_jolo(input_csv):
    # Columns to complete
    exam_cols = [
        "tmt_a",
        "tmt_a_chyby",
        "tmt_b",
        "tmt_b_chyby",
        "jolo"
    ]

    # Read data
    df = pd.read_csv(input_csv)

    # Sort to ensure enrollment comes before follow-up
    df = df.sort_values(["study_id", "redcap_event_name"])

    # Process each patient
    for study_id, group in df.groupby("study_id"):
        if len(group) < 2:
            continue

        enrollment = group[group["redcap_event_name"].str.contains("enrollment", na=False)]
        followup = group[group["redcap_event_name"].str.contains("followup", na=False)]

        if enrollment.empty or followup.empty:
            continue

        enroll_idx = enrollment.index[0]
        follow_idx = followup.index[0]

        for col in exam_cols:
            if pd.isna(df.loc[follow_idx, col]) and not pd.isna(df.loc[enroll_idx, col]):
                df.loc[follow_idx, col] = df.loc[enroll_idx, col]

    # Output file name
    base, ext = os.path.splitext(input_csv)
    output_csv = f"{base}_completed-tmt-jolo{ext}"

    # Save
    df.to_csv(output_csv, index=False)

    print(f"Completed file saved as: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill missing TMT/Jolo values in follow-up rows using enrollment values"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )

    args = parser.parse_args()
    complete_tmt_jolo(args.input_csv)
