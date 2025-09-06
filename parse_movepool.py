# Script to parse movepools from the gen9randombattle.json data
import json
import joblib
from pathlib import Path
import requests

def fetch_and_cache_json(
    url: str,
    cache_path: Path
) -> dict:
    # downloads json if doesnt exist already
    if not cache_path.exists():
        print(f"Downloading {url} → {cache_path}")
        resp = requests.get(url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"Loading cached JSON from {cache_path}")

    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_full_movepool(
    data: dict
) -> dict[str, set[str]]:
    # map each species to its full possible movepool
    movepools: dict[str, set[str]] = {}
    for species, info in data.items():
        moveset = set()
        roles = info.get("roles", {})
        for role_name, role_data in roles.items():
            for mv in role_data.get("moves", []):
                moveset.add(mv)
        movepools[species] = moveset
    return movepools

def save_movepool(
    movepools: dict[str, set[str]],
    out_path: Path
):
    # save the movepool using joblib
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(movepools, out_path)
    print(f"Saved movepools ({len(movepools)} species) → {out_path}")

if __name__ == "__main__":
    URL = "https://pkmn.github.io/randbats/data/gen9randombattle.json"
    CACHE_JSON = Path("data/gen9randombattle.json")
    OUT_PICKLE = Path("data/processed/full_movepools.pkl")

    raw = fetch_and_cache_json(URL, CACHE_JSON)

    movepools = build_full_movepool(raw)

    save_movepool(movepools, OUT_PICKLE)