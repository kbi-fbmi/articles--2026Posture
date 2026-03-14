import pandas as pd
import sys
from pathlib import Path

# =========================
# Arguments
# =========================
main_file = sys.argv[1]
duration_file = sys.argv[2]

# =========================
# Load data
# =========================
df_main = pd.read_csv(main_file)
df_duration = pd.read_csv(duration_file)

# =========================
# Build lookup from second file
# =========================
duration_lookup = (
    df_duration
    .set_index(["study_id", "redcap_event_name"])["pd_duration"]
)

# =========================
# Fill duration_pd in main file
# =========================
df_main["duration_pd"] = df_main.set_index(
    ["study_id", "redcap_event_name"]
).index.map(duration_lookup)

# =========================
# Restore index
# =========================
df_main.reset_index(drop=True, inplace=True)

# =========================
# Build output filename (updated)
# =========================
input_path = Path(main_file)
output_file = input_path.stem + "_duration" + input_path.suffix

# =========================
# Save output
# =========================
df_main.to_csv(output_file, index=False)

print(f"Created: {output_file}")
