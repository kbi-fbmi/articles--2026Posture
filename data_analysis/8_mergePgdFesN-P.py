import pandas as pd
import sys
from pathlib import Path

# =========================
# Arguments
# =========================
input_file = sys.argv[1]

# =========================
# Load data
# =========================
df = pd.read_csv(input_file)

# =========================
# Merge RBD columns into main columns
# =========================
df["pdq_soucet"] = df["pdq_soucet"].combine_first(df["pdq_soucet_rbd"])
df["fes_score"] = df["fes_score"].combine_first(df["fes_score_rbd"])

# =========================
# Remove RBD columns
# =========================
df.drop(columns=["pdq_soucet_rbd", "fes_score_rbd"], inplace=True)

# =========================
# Compute time difference N − P
# =========================
df["time_meanP-meanN"] = (
    df["w_ex_time_meanP1P2"] - df["w_ex_time_meanN1N2"]
)

# =========================
# Build output filename
# =========================
input_path = Path(input_file)
output_file = (
    input_path.stem
    + "_pdg_merged_fes_merged_P-N"
    + input_path.suffix
)

# =========================
# Save output
# =========================
df.to_csv(output_file, index=False)

print(f"Created: {output_file}")
