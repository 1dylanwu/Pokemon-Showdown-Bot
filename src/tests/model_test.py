import joblib
import numpy as np
from src.bot.decision_model import DecisionModel

def main():
    model = DecisionModel()
    state = {
        "turn": 8,
        "side": "p1a",
        
        "p1a_active": "ursaring",
        "p2a_active": "forretress", 
        
        "p1a_hp_pct": 0.05,
        "p2a_hp_pct": 0.8,
        "p1a_status": "none",
        "p2a_status": "none",
        "p1a_fainted": 0,
        "p2a_fainted": 0,
        
        "p1a_is_terastallized": 0,
        "p2a_is_terastallized": 0,
        "p1a_tera_type": "none",
        "p2a_tera_type": "none",
        
        "p1_team_species": ["cinderace", "ursaring", "copperajah", "primarina", "toucannon"],
        "p2_team_species": ["alcremie", "altaria", "forretress", "keldeoresolute", "ironleaves", "screamtail"],
        "p1_available": ["cinderace", "copperajah", "primarina", "toucannon"],
        "p2_available": ["alcremie", "altaria", "keldeoresolute", "ironleaves", "screamtail"],
        
        "weather": "clear",
        "terrain": "none",

        "p1_type_matchup": 1.5,
        "p2_type_matchup": 1.5,
        
        "hazards_p1": {},
        "hazards_p2": {},

        "p1a_boost_atk": 0,
        "p1a_boost_def": 0,
        "p1a_boost_spa": 0,
        "p1a_boost_spd": 0,
        "p1a_boost_spe": 0,
        "p2a_boost_atk": 0,
        "p2a_boost_def": 0,
        "p2a_boost_spa": 0,
        "p2a_boost_spd": 0,
        "p2a_boost_spe": 0,

        "p1_known_hp_cinderace": 1.0,
        "p1_known_hp_ursaring": 1.0,
        "p1_known_hp_copperajah": 0.1,
        "p1_known_hp_primarina": 1.0,
        "p1_known_hp_toucannon": 1.0,
        #"p1_known_hp_alakazam": 100.0,
        
        "p2_known_hp_alcremie": 1.0,
        "p2_known_hp_altaria": 1.0,
        "p2_known_hp_forretress": 1.0,
        "p2_known_hp_keldeoresolute": 1.0,
        "p2_known_hp_ironleaves": 1.0,
        "p2_known_hp_screamtail": 1.0,
    }
    res = model.predict(state, {"bodyslam", "throatchop", "rest", "sleeptalk"}, {"cinderace", "copperajah", "primarina", "toucannon"})
    print(res["action_type"])
    print(res["action"])
    print(res["confidence"])

if __name__ == "__main__":
    main()