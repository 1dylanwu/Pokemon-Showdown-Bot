import json
from pathlib import Path
from collections import defaultdict
import joblib
import pandas as pd

CSV_FILES = [
    Path("data/parsed/train.csv"),
    Path("data/parsed/val.csv"),
    Path("data/parsed/test.csv")
]
OUT_PATH = Path("data/processed/move/full_movepool.pkl")
JSON_PATH = Path("data/raw/gen9randombattle.json")

with open(JSON_PATH) as f:
    role_data = json.load(f)

movepools = defaultdict(set)

for csv_path in CSV_FILES:
    df = pd.read_csv(csv_path, dtype=str)
    df = df[df["action_type"] == "move"]
    for _, row in df.iterrows():
        species = row["state_p1a_active"] if row["side"] == "p1a" else row["state_p2a_active"]
        move = row["action"]
        movepools[species.strip()].add(move.strip())

for species, data in role_data.items():
    for role in data.get("roles", {}).values():
        for mv in role.get("moves", []):
            movepools[species.strip()].add(mv.strip())


movepools = {sp: sorted(list(moves)) for sp, moves in movepools.items()}
joblib.dump(movepools, OUT_PATH)
print(f"Saved movepools for {len(movepools)} species to {OUT_PATH}")
