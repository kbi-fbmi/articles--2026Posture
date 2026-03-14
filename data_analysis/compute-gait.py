import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python3 compute-gait.py input.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = "merged_gait_summary.csv"

df = pd.read_csv(input_file)

for col in ["w_ex_gr_cad_r", "w_ex_gr_velo_l"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

final_rows = []

grouped = df.groupby(["study_id", "redcap_event_name"], sort=False)

for (sid, event), block in grouped:

    block = block.copy()

    # identify summary row
    summary_mask = block["redcap_repeat_instrument"].isna()
    summary_rows = block[summary_mask]

    if summary_rows.empty:
        # safety fallback
        summary_row = block.iloc[0].copy()
    else:
        summary_row = summary_rows.iloc[0].copy()

    # gait experiment rows
    gait_rows = block[block["redcap_repeat_instrument"] == "walk_exam"]

    gait_rows = gait_rows[
        gait_rows["w_ex_gr_cad_r"].notna() |
        gait_rows["w_ex_gr_velo_l"].notna()
    ]

    gait_rows = gait_rows.head(2)

    if not gait_rows.empty:
        summary_row["w_ex_gr_cad_r"] = gait_rows["w_ex_gr_cad_r"].mean()
        summary_row["w_ex_gr_velo_l"] = gait_rows["w_ex_gr_velo_l"].mean()

    final_rows.append(summary_row)

final_df = pd.DataFrame(final_rows)
final_df.to_csv(output_file, index=False)

print("✅ Finished correctly")
print("Output file:", output_file)
