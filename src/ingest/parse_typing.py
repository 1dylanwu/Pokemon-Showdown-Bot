import json
import requests
import time
from pathlib import Path

def fetch_types(poke_name):
    url = "https://pokeapi.co/api/v2/pokemon/" + poke_name.lower()
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return [t["type"]["name"] for t in resp.json()["types"]]
    except Exception as e:
        print(f"Failed to fetch {poke_name}: {e}")
        return []

def main():
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

if __name__ == "__main__":
    main()