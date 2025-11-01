from pathlib import Path
import csv
import json
from typing import Dict, Any
from src.utils.utils import normalize
import pandas as pd


def load_base_stats(csv_path: Path | str) -> Dict[str, Dict[str, float]]:
    """
    Load base stats CSV with columns: species,hp,atk,def,spa,spd,spe
    Returns mapping keyed by normalized species name.
    """
    csv_path = Path(csv_path)
    out: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = normalize(row.get("Name"))
            out[sp] = {
                "hp": float(row.get("hp")),
                "atk": float(row.get("atk")),
                "def": float(row.get("def")),
                "spa": float(row.get("spa")),
                "spd": float(row.get("spd")),
                "spe": float(row.get("spe")),
            }
    return out

def compute_pokemon_stats(
    species: str,
    level: int,
    base_stats: Dict[str, Dict[str, float]],
    ivs: Dict[str, int] | None = None,
    evs: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """
    Compute actual Pokemon stats at given level.
    By default, IVs are 31 and EVs are 84.
    Returns integer stats: hp, atk, def, spa, spd, spe.
    Formulas:
      HP: floor(((2*Base + IV + EV/4) * Level) / 100) + Level + 10
      Other: floor((floor(((2*Base + IV + EV/4) * Level) / 100) + 5) * Nature)
    """
    norm = normalize(species)
    if norm.startswith("arceus"):
        norm = "arceus"
    base = base_stats.get(norm)

    # default IVs and EVs
    if ivs is None:
        ivs = {k: 31 for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    else:
        ivs = {k: int(ivs.get(k, 31)) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    if evs is None:
        evs = {k: 84 for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    else:
        evs = {k: int(evs.get(k, 84)) for k in ("hp", "atk", "def", "spa", "spd", "spe")}

    def calc_hp(base_hp, iv, ev, lvl):
        return int(((2 * base_hp + iv + (ev // 4)) * lvl) // 100 + lvl + 10)

    def calc_stat(base_stat, iv, ev, lvl):
        return int(((2 * base_stat + iv + (ev // 4)) * lvl) // 100 + 5)

    lvl = int(level)
    stats = {}
    stats["hp"] = calc_hp(base["hp"], ivs["hp"], evs["hp"], lvl)
    stats["atk"] = calc_stat(base["atk"], ivs["atk"], evs["atk"], lvl)
    stats["def"] = calc_stat(base["def"], ivs["def"], evs["def"], lvl)
    stats["spa"] = calc_stat(base["spa"], ivs["spa"], evs["spa"], lvl)
    stats["spd"] = calc_stat(base["spd"], ivs["spd"], evs["spd"], lvl)
    stats["spe"] = calc_stat(base["spe"], ivs["spe"], evs["spe"], lvl)
    return stats

def load_randbat_json(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_stats_table(randbat_json_path: Path | str, base_stats_csv: Path | str) -> Dict[str, Dict[str,int]]:
    """
    Reads randbat JSON and base stats CSV to compute actual stats for each species.
    randbat JSON has level, and optionally ivs and evs for each species.
    Returns mapping: species_name -> {level: int, hp: int, atk:int, def:int, spa:int, spd:int, spe:int}
    """
    base_stats = load_base_stats(Path(base_stats_csv))
    data = load_randbat_json(Path(randbat_json_path))
    out = {}
    for species_name, info in data.items():
        level = int(info.get("level"))
        evs = info.get("evs", {}) or {}
        ivs = info.get("ivs", {}) or {}
        # compute stats
        stats = compute_pokemon_stats(species_name, level, base_stats, ivs=ivs, evs=evs)
        out[normalize(species_name)] = {"level": level, **stats}
    return out

if __name__ == "__main__":
    randbat_path = Path("data/raw/gen9randombattle.json")
    base_stats_path = Path("data/raw/raw_stats.csv")
    stats_table = build_stats_table(randbat_path, base_stats_path)

    df = pd.DataFrame.from_dict(stats_table, orient="index")
    df.index.name = "species"
    df.reset_index().to_csv("data/raw/computed_stats.csv", index=False)
    with open("data/raw/computed_stats.json", "w") as f:
        json.dump(stats_table, f, indent=2)