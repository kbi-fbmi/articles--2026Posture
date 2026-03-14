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
# Identify enrollment and follow-up rows
# =========================
is_enrollment = df["redcap_event_name"].isin(
    ["enrollment_arm_1", "enrollment_arm_2"]
)

is_followup = df["redcap_event_name"].str.startswith(
    "followup", na=False
)

# =========================
# Build enrollment date lookup (BEFORE filtering)
# =========================
enrollment_date_lookup = (
    df.loc[is_enrollment]
    .set_index("study_id")["datum_vysetreni"]
)

# =========================
# Find patients with follow-up
# =========================
patients_with_followup = set(
    df.loc[is_followup, "study_id"]
)

# =========================
# Remove enrollment rows IF patient has follow-up
# =========================
df = df[
    ~(
        is_enrollment
        & df["study_id"].isin(patients_with_followup)
    )
].copy()

# =========================
# Add enrollment date column to ALL remaining rows
# =========================
df["datum_vysetreni_Enrollment"] = (
    df["study_id"].map(enrollment_date_lookup)
)

# =========================
# Reorder columns (insert after datum_vysetreni)
# =========================
cols = list(df.columns)
idx = cols.index("datum_vysetreni")

cols.insert(idx + 1, cols.pop(cols.index("datum_vysetreni_Enrollment")))
df = df[cols]

# =========================
# Save output
# =========================
output_file = "Final_data_frame_filtered_enrollment.csv"
df.to_csv(output_file, index=False)

print(f"Created: {output_file}")
