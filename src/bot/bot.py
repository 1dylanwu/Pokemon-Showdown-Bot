from typing import Dict
from poke_env.player.player import Player
from poke_env.battle import Battle
from src.bot.decision_model import DecisionModel
from src.ingest.add_type_features import type_effectiveness, get_pokemon_types
from src.ingest.parse_replays import COSMETIC_SPECIES

class AIAgent(Player):
    def __init__(
        self,
        model,
        battle_format="gen9randombattle",
        server_configuration=None,
        team=None,
        max_concurrent_battles=1
    ):
        super().__init__(
            battle_format=battle_format,
            server_configuration=server_configuration,
            team=team,
            max_concurrent_battles=max_concurrent_battles
        )
        self.model = model
    
    def choose_move(self, battle):
        state = self.format_state(battle)

        # 2) Extract legal move IDs and switch species names
        legal_moves = [move.id for move in battle.available_moves]
        legal_switches = [self.normalize(p.species) for p in battle.available_switches]

        if battle.force_switch:
            decision = self.model.predict_forced(state, legal_switches)
            for p in battle.available_switches:
                if self.normalize(p.species) == decision["action"]:
                    return self.create_order(p)

        decision = self.model.predict(state, legal_moves, legal_switches)
        if decision["action_type"] == "move" or battle.trapped:
            # find the matching Move object
            for move in battle.available_moves:
                if move.id == decision["action"][5:]:
                    return self.create_order(move)

            # fallback if something went wrong
            print(f"couldnt find move {decision["action"]}")
            return self.choose_random_move(battle)
        else:
            for p in battle.available_switches:
                if self.normalize(p.species) == decision["action"]:
                    return self.create_order(p)
        
        print("wow i suck")
        return self.choose_random_move(battle)
        # either switch or forced switch

    def normalize(self, name: str):
        for species in COSMETIC_SPECIES:
            if name.startswith(species.lower()):
                return species.lower()
        return name.lower()

    def format_state(self, battle) -> Dict[str, any]:
        """
        Build a full feature dict from a poke-env Battle object,
        including active Pokémon, bench, hazards, boosts, types, and matchup.
        """
        # You are always p1 in your own battle object
        side = "p1a"
        opp_side = "p2a"

        # Your vs. opponent active Pokémon
        p1a = battle.active_pokemon
        p2a = battle.opponent_active_pokemon

        # Your legal switches (poke-env provides this)
        p1_switches = battle.available_switches

        # Approximate opponent’s legal switches:
        # any revealed, non-fainted, non-active mon
        p2_switches = [
            mon for mon in battle.opponent_team.values()
            if mon.species != "???" and not mon.fainted and not mon.active
        ]


        # Team species lists
        p1_team = [self.normalize(p1a.species)] + [self.normalize(m.species) for m in p1_switches]
        p2_team = [self.normalize(p2a.species)] + [self.normalize(m.species) for m in p2_switches]

        # Available switch species
        p1_avail = [self.normalize(m.species) for m in p1_switches]
        p2_avail = [self.normalize(m.species) for m in p2_switches]

        # Count fainted (assuming team size = 6)
        team_size = 6
        p1_fainted = team_size - len(p1_team)
        p2_fainted = team_size - len(p2_team)

        # Weather & terrain
        weather = next(iter(battle.weather), "clear")
        terrain = battle.fields.get("terrain", "none")

        # Hazards
        hazards_p1 = battle.side_conditions.get(side, {})
        hazards_p2 = battle.side_conditions.get(opp_side, {})

        # Known HP map (use the same switch lists)
        known_hp_map = {}
        for mon in [p1a] + p1_switches + [p2a] + p2_switches:
            known_hp_map[self.normalize(mon.species)] = mon.current_hp_fraction

        # Types from your JSON—or replace with p1a.type_1 / type_2 if preferred
        p1a_types = get_pokemon_types(self.normalize(p1a.species))
        p2a_types = get_pokemon_types(self.normalize(p2a.species))
        p1a_tera = getattr(p1a, "tera_type", "none")
        p2a_tera = getattr(p2a, "tera_type", "none")

        # Compute matchup scores
        p1_type_matchup = type_effectiveness(
            attacking_types=p1a_types,
            defending_types=p2a_types,
            tera_type=p1a_tera,
        )
        p2_type_matchup = type_effectiveness(
            attacking_types=p2a_types,
            defending_types=p1a_types,
            tera_type=p2a_tera,
        )

        # Assemble the state dict
        state = {
            "turn": battle.turn,
            "side": side,

            # Active Pokémon
            "p1a_active": self.normalize(p1a.species),
            "p2a_active": self.normalize(p2a.species),
            "p1a_hp_pct": p1a.current_hp_fraction,
            "p2a_hp_pct": p2a.current_hp_fraction,
            "p1a_status": p1a.status.name if p1a.status else "none",
            "p2a_status": p2a.status.name if p2a.status else "none",
            "p1a_fainted": p1_fainted,
            "p2a_fainted": p2_fainted,

            # Terastalization
            "p1a_is_terastallized": int(getattr(p1a, "is_terastallized", False)),
            "p2a_is_terastallized": int(getattr(p2a, "is_terastallized", False)),
            "p1a_tera_type": p1a_tera,
            "p2a_tera_type": p2a_tera,

            # Types & matchup
            "p1a_types": p1a_types,
            "p2a_types": p2a_types,
            "p1_type_matchup": p1_type_matchup,
            "p2_type_matchup": p2_type_matchup,

            # Teams and legal switches
            "p1_team_species": p1_team,
            "p2_team_species": p2_team,
            "p1_available": p1_avail,
            "p2_available": p2_avail,

            # Field conditions
            "weather": weather,
            "terrain": terrain,

            # Hazards
            "hazards_p1": hazards_p1,
            "hazards_p2": hazards_p2,
        }

        # Stat boosts
        for stat in ("atk", "def", "spa", "spd", "spe"):
            state[f"p1a_boost_{stat}"] = p1a.boosts.get(stat, 0)
            state[f"p2a_boost_{stat}"] = p2a.boosts.get(stat, 0)

        # Known HP for every species on both teams
        for species in set(p1_team + p2_team):
            hp = known_hp_map.get(species, 1.0)
            state[f"p1_known_hp_{species}"] = hp
            state[f"p2_known_hp_{species}"] = hp

        return state
