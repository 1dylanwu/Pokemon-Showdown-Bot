import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Dict, List, Tuple
import json

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
    return POKEMON_TYPES.get(species, ['Normal'])  # default to normal if not found

def type_effectiveness(attacking_types: List[str], defending_types: List[str], tera_type: str) -> float:
    if not attacking_types or not defending_types:
        return 1.0
    # computes the best type effectiveness of one of the attacking types
    total_effectiveness = 0
    tera_type = str(tera_type).strip().capitalize() if pd.notna(tera_type) else "None"
    for attack_type in attacking_types:
        temp = 1.0
        for defense_type in defending_types:
            temp *= TYPE_EFFECTIVENESS.get(attack_type, {}).get(defense_type, 1.0)
        # if its tera the stab mult is 2 instead of 1.5
        if attack_type == tera_type and attack_type in attacking_types:
            temp *= 2
        else:
            temp *= 1.5
        total_effectiveness = max(total_effectiveness, temp)
    
    return total_effectiveness

def add_type_features(df: pd.DataFrame) -> pd.DataFrame:
    global POKEMON_TYPES
    
    # get active pokemon types
    df['p1a_types'] = df['state_p1a_active'].apply(get_pokemon_types)
    df['p2a_types'] = df['state_p2a_active'].apply(get_pokemon_types)

    def defending_types(row, side: str) -> List[str]:
        tera_raw = row.get(f"state_{side}_tera_type", "none")
        tera = str(tera_raw).strip().capitalize() if pd.notna(tera_raw) else "None"
        is_tera = row[f"state_{side}_is_terastallized"] == "True"
        if is_tera and tera != "None":
            # defense is purely the tera type when terastallized
            return [tera]
        return get_pokemon_types(row[f"state_{side}_active"])

    # calculate type matchups for each pokemon's offensive types
    df['p1_type_matchup'] = df.apply(
        lambda r: type_effectiveness(
            attacking_types = get_pokemon_types(r["state_p1a_active"]),
            defending_types = defending_types(r, "p2a"),
            tera_type = r["state_p1a_tera_type"]
        ),
        axis=1
    )
    df['p2_type_matchup'] = df.apply(
        lambda r: type_effectiveness(
            attacking_types = get_pokemon_types(r["state_p2a_active"]),
            defending_types = defending_types(r, "p1a"),
            tera_type = r["state_p2a_tera_type"]
        ),
        axis=1
    )
    
    return df

if __name__ == "__main__":
    master_path = Path("data/parsed/master.csv")
    train_path = Path("data/parsed/train.csv")
    val_path = Path("data/parsed/val.csv")
    test_path = Path("data/parsed/test.csv")
    
    for path in [master_path, train_path, val_path, test_path]:
        df = pd.read_csv(path, dtype=str)
        df_plus = add_type_features(df)
        df_plus.to_csv(path, index = False)
