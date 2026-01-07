import ast
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Dict, List, Tuple
import json
from src.utils.utils import normalize

# Type effectiveness chart (Gen 9)
TYPE_EFFECTIVENESS = {
    'Normal': {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
    'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
    'Water': {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 'Rock': 2, 'Dragon': 0.5},
    'Electric': {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0, 'Flying': 2, 'Dragon': 0.5},
    'Grass': {'Fire': 0.5, 'Water': 2, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2, 'Dragon': 0.5, 'Steel': 0.5},
    'Ice': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 0.5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': 0.5},
    'Fighting': {'Normal': 2, 'Ice': 2, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dark': 2, 'Steel': 2, 'Fairy': 0.5},
    'Poison': {'Grass': 2, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0, 'Fairy': 2},
    'Ground': {'Fire': 2, 'Electric': 2, 'Grass': 0.5, 'Poison': 2, 'Flying': 0, 'Bug': 0.5, 'Rock': 2, 'Steel': 2},
    'Flying': {'Electric': 0.5, 'Grass': 2, 'Ice': 0.5, 'Fighting': 2, 'Bug': 2, 'Rock': 0.5, 'Steel': 0.5},
    'Psychic': {'Fighting': 2, 'Poison': 2, 'Psychic': 0.5, 'Dark': 0, 'Steel': 0.5},
    'Bug': {'Fire': 0.5, 'Grass': 2, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2, 'Ghost': 0.5, 'Dark': 2, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock': {'Fire': 2, 'Ice': 2, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2, 'Bug': 2, 'Steel': 0.5},
    'Ghost': {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5},
    'Dragon': {'Dragon': 2, 'Steel': 0.5, 'Fairy': 0},
    'Dark': {'Fighting': 0.5, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5, 'Fairy': 0.5},
    'Steel': {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2, 'Rock': 2, 'Steel': 0.5, 'Fairy': 2},
    'Fairy': {'Fire': 0.5, 'Fighting': 2, 'Poison': 0.5, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5}
}

POKEMON_TYPES = json.loads(Path("data/raw/poke_types.json").read_text())

def get_pokemon_types(species: str) -> List[str]:
    # get the types of a pokemon species
    return POKEMON_TYPES.get(normalize(species))

def _parse_species_cell(cell) -> List[str]:
    # assume cell is a string like "['Tinkaton', 'Dachsbun']"
    return [s.strip() for s in ast.literal_eval(cell)]

def type_effectiveness(attacking_types: List[str], defending_types: List[str], tera_type: str) -> float:
    # computes the best type effectiveness of one of the attacking types
    total_effectiveness = 0
    tera_type = str(tera_type).strip().capitalize() if pd.notna(tera_type) else "None"
    for attack_type in attacking_types:
        temp = 1.0
        for defense_type in defending_types:
            temp *= TYPE_EFFECTIVENESS.get(attack_type, {}).get(defense_type, 1.0)
        # if its tera the stab mult is 2 instead of 1.5
        if attack_type == tera_type:
            temp *= 2
        else:
            temp *= 1.5
        total_effectiveness = max(total_effectiveness, temp)
    
    return total_effectiveness

def compute_row(r):
    side = str(r.get("side", "")).strip().lower()

    # determine own active and available switches (parsed), excluding the active
    if side == "p1":
        own_active = r.get("p1_active")
        avail_switch = _parse_species_cell(r.get("p1_team_species"))
        opp = r.get("p2_active")
    elif side == "p2":
        own_active = r.get("p2_active")
        avail_switch = _parse_species_cell(r.get("p2_team_species"))
        opp = r.get("p1_active")
    else:
        print(f"Unknown side: {side}")

    # remove the current active from the available switches
    norm_active = normalize(own_active)
    filtered_avail = [s for s in avail_switch if normalize(s) != norm_active]

    opp_types = get_pokemon_types(opp)
    if not opp_types:
        print(f"Missing opponent types for: {opp}")
        opp_types = []

    best_off = 0.0
    best_def = 7.0

    for sp in filtered_avail:
        sp_types = get_pokemon_types(sp)
        if not sp_types:
            print(sp)
            continue
        off = type_effectiveness(sp_types, opp_types, "None")
        best_off = max(best_off, off)
        deff = type_effectiveness(opp_types, sp_types, "None")
        best_def = min(best_def, deff)
    
    return pd.Series({
        "best_offensive_matchup": best_off,
        "best_defensive_mathcup": best_def
    })

def add_type_features(df: pd.DataFrame) -> pd.DataFrame:
    global POKEMON_TYPES
    
    # get active pokemon types
    df['p1_types'] = df['p1_active'].apply(get_pokemon_types)
    df['p2_types'] = df['p2_active'].apply(get_pokemon_types)

    def defending_types(row, side: str) -> List[str]:
        tera_raw = row.get(f"{side}_tera_type", "none")
        tera = str(tera_raw).strip().capitalize() if pd.notna(tera_raw) else "None"
        is_tera = row[f"{side}_is_terastallized"] == "True"
        if is_tera and tera != "None":
            # defense is purely the tera type when terastallized
            return [tera]
        return get_pokemon_types(row[f"{side}_active"])

    # calculate type matchups for each pokemon's offensive types
    df['p1_type_matchup'] = df.apply(
        lambda r: type_effectiveness(
            attacking_types = get_pokemon_types(r["p1_active"]),
            defending_types = defending_types(r, "p2"),
            tera_type = r["p1_tera_type"]
        ),
        axis=1
    )
    df['p2_type_matchup'] = df.apply(
        lambda r: type_effectiveness(
            attacking_types = get_pokemon_types(r["p2_active"]),
            defending_types = defending_types(r, "p1"),
            tera_type = r["p2_tera_type"]
        ),
        axis=1
    )
    
    return df

if __name__ == "__main__":
    train_path = Path("data/parsed/train.csv")
    val_path = Path("data/parsed/val.csv")
    test_path = Path("data/parsed/test.csv")
    
    for path in [train_path, val_path, test_path]:
        df = pd.read_csv(path, dtype=str)
        #matchup_df = df.apply(compute_row, axis=1)
        #df = pd.concat([df, matchup_df], axis=1)
        output_path = path.with_name(path.stem + "a.csv")
        df.to_csv(output_path, index=False)