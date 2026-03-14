import pandas as pd
import argparse
import os
import re

# Patients requiring re-classification
OVERRIDE_PATIENTS = {
    "RBD017-c", "RBD022-c", "RBD038-c", "RBD050-c",
    "RBD061-c", "RBD073-c", "RBD108-c", "RBD113-c",
    "RBD123-c"
}


def extract_followup_number(event_name):
    if isinstance(event_name, str):
        match = re.search(r"followup_(\d+)", event_name)
        if match:
            return int(match.group(1))
    return -1


def classify_td_pigd(tremor, pigd):
    if pd.isna(tremor) or pd.isna(pigd):
        return "PD_UNCLASSIFIED"

    if tremor > 0 and pigd == 0:
        return "TD"
    if tremor == 0 and pigd > 0:
        return "PIGD"

    if pigd > 0:
        ratio = tremor / pigd
        if ratio >= 1.15:
            return "TD"
        if ratio <= 0.9:
            return "PIGD"

    return "PD_UNCLASSIFIED"


def transfer_last_group_with_override(source_csv, target_csv):
    df_source = pd.read_csv(source_csv)
    df_target = pd.read_csv(target_csv)

    # Determine follow-up order
    df_source["followup_order"] = df_source["redcap_event_name"].apply(extract_followup_number)

    # Keep rows with group or UPDRS data
    df_source_valid = df_source.dropna(
        subset=["mdsupdrs_td", "mdsupdrs_pigd"], how="all"
    )

    # Select last follow-up per patient
    last_rows = (
        df_source_valid
        .sort_values("followup_order")
        .groupby("study_id")
        .tail(1)
    )

    group_map = {}

    for _, row in last_rows.iterrows():
        study_id = row["study_id"]

        if study_id in OVERRIDE_PATIENTS:
            group_map[study_id] = classify_td_pigd(
                row.get("mdsupdrs_td"),
                row.get("mdsupdrs_pigd")
            )
        else:
            group_map[study_id] = row.get("group", "PD_UNCLASSIFIED")

    # Fill target file
    df_target["group"] = df_target["study_id"].map(group_map)

    # Output
    base, ext = os.path.splitext(target_csv)
    output_csv = f"{base}_group-last-followup-reclassified{ext}"
    df_target.to_csv(output_csv, index=False)

    print("Group transfer completed.")
    print(f"Output file: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer group from last follow-up and reclassify specific patients using TD/PIGD ratio rules"
    )
    parser.add_argument("--source", required=True, help="Source longitudinal CSV")
    parser.add_argument("--target", required=True, help="Target CSV to update group column")

    args = parser.parse_args()
    transfer_last_group_with_override(args.source, args.target)
