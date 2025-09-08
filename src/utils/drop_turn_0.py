import pandas as pd
from pathlib import Path

INPUT_CSV = Path("data/parsed/val.csv")
OUTPUT_CSV = Path("data/parsed/val0.csv")

df = pd.read_csv(INPUT_CSV, dtype=str)

df["turn"] = pd.to_numeric(df["turn"], errors="coerce")

df_cleaned = df[df["turn"] > 0].copy()

df_cleaned.to_csv(OUTPUT_CSV, index=False)

print(f"Removed turn 0 rows: {len(df) - len(df_cleaned)} dropped")
print(f"Saved cleaned CSV to: {OUTPUT_CSV}")