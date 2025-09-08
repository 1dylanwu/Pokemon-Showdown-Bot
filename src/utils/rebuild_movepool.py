import json
from pathlib import Path
from collections import defaultdict
import joblib
import pandas as pd

CSV_DIR = Path("data/parsed") 
JSON_PATH = Path("data/raw/gen9randombattle.json") 
OUT_PATH = Path("data/processed/move/full_movepools_from_data.pkl")

with JSON_PATH.open() as f:
    role_data = json.load(f)

def extract_from_df(df: pd.DataFrame, movepools: dict[str,set]):
    # keep only actual moves
    df = df[df["action_type"] == "move"]
    if "state_p1a_active" in df.columns:
        df.rename(columns=lambda c: c[6:] if c.startswith("state_") else c, inplace=True)
    # pull species+move per row
    for sp1, sp2, player, move_label in zip(
        df["p1a_active"], df["p2a_active"], df["side"], df["action"]
    ):
        # raw move
        raw = move_label.split("_", 1)[1] if "_" in move_label else move_label
        # active species depends on player
        species = sp1 if player == "p1" else sp2
        movepools[species].add(raw)

movepools = defaultdict(set)

# from all CSVs (e.g. train.csv, val.csv)
for csv_file in CSV_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file, dtype=str)
    df = df[df["action_type"] == "move"].copy()
    extract_from_df(df, movepools)

for species, spec_data in role_data.items():
    roles = spec_data.get("roles", {})
    for role_name, role_info in roles.items():
        moves = role_info.get("moves", [])
        for move in moves:
            movepools[species].add(move)

movepools = {sp: sorted(list(moves)) for sp, moves in movepools.items()}

joblib.dump(movepools, OUT_PATH)
print(f"Extracted movepools for {len(movepools)} species → saved to {OUT_PATH}")

