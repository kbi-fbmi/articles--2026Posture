import pandas as pd
import sys

# =========================
# Arguments
# =========================
input_file = sys.argv[1]

# =========================
# Load data
# =========================
df = pd.read_csv(input_file)

# =========================
# Enrollment dataframe
# =========================
df_enrollment = df[
    df["redcap_event_name"].isin(["enrollment_arm_1", "enrollment_arm_2"])
].copy()

df_enrollment.to_csv(
    "Enrolment_final_data_frame.csv",
    index=False
)

# =========================
# Follow-up dataframe
# =========================
df_followup = df[df["redcap_event_name"].str.startswith("followup", na=False)].copy()

# =========================
# Build enrollment date lookup
# =========================
enrollment_date_lookup = (
    df_enrollment
    .set_index("study_id")["datum_vysetreni"]
)

# =========================
# Insert datum_vysetreni_Enrollment column
# =========================
df_followup["datum_vysetreni_Enrollment"] = (
    df_followup["study_id"].map(enrollment_date_lookup)
)

# =========================
# Reorder columns (insert after datum_vysetreni)
# =========================
cols = list(df_followup.columns)
idx = cols.index("datum_vysetreni")

cols.insert(idx + 1, cols.pop(cols.index("datum_vysetreni_Enrollment")))
df_followup = df_followup[cols]

# =========================
# Save follow-up output
# =========================
df_followup.to_csv(
    "Followup_final_data_frame.csv",
    index=False
)

print("Created:")
print(" - Followup_final_data_frame.csv")
print(" - Enrolment_final_data_frame.csv")
