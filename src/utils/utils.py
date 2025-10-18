from ast import Set
import numpy as np
import json
import time
from pathlib import Path
import pandas as pd
import requests

def normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")

def split_action_type(y):
    damage_moves = set()
    with open("data/raw/moves.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                damage_moves.add(normalize(line.strip()))
    types = []
    for act in y:
        if act.startswith("forced_switch_"):
            types.append("forced")
        elif act.startswith("switch_"):
            types.append("switch")
        elif act.startswith("move_"):
            move = act[len("move_"):]
            move = normalize(move)
            if move in damage_moves:
                types.append("move_damage")
            else:
                types.append("move_utility")
        else:
            types.append("unknown")
    return np.array(types, dtype=str)


def save_poke_types():
    def fetch_types(poke_name):
        url = "https://pokeapi.co/api/v2/pokemon/" + poke_name.lower()
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return [t["type"]["name"] for t in resp.json()["types"]]
        except Exception as e:
            print(f"Failed to fetch {poke_name}: {e}")
            return []
        
    data = json.loads(Path("data/raw/gen9randombattle.json").read_text())
    poke_names = list(data.keys())
    print(len(poke_names))
    mapping = {}
    for name in poke_names:
        if name.lower().startswith("arceus-"):
            mapping[name] = name.split("-")[1].lower()
            continue
        types = fetch_types(name)
        if types:
            mapping[name] = types
        time.sleep(0.1)
    Path("data/raw/poke_types.json").write_text(json.dumps(mapping, indent=2))

def filter_by_turn_percentile(csv_path: str,
                              out_path: str = None,
                              percentile: float = 99.0,
                              turn_col: str = "turn"):
    df = pd.read_csv(csv_path)

    cutoff = df[turn_col].quantile(percentile / 100.0)
    cutoff_int = int(cutoff) if cutoff.is_integer() else cutoff
    print(f"{percentile}th percentile cutoff for '{turn_col}': {cutoff_int}")

    filtered = df[df[turn_col] <= cutoff]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_path, index=False)
        print(f"Filtered data saved to: {out_path}")

    return filtered, cutoff