from poke_env import Player
from poke_env.battle import Move, Pokemon, Battle
from poke_env.calc import calculate_damage, calculate_base_power
from type_chart import get_effectiveness

class RulesBot(Player):
    def choose_move(self, battle: Battle):

        #if we have bad defensive type matchup, switch out
        defensive_switch = False
        for enemy_type in battle.opponent_active_pokemon.types:
            #checks enemy STAB typing against our active pokemon
            if get_effectiveness(enemy_type, battle.active_pokemon.types) > 1.0:
                defensive_switch = True

        if defensive_switch:
            #find a pokemon to switch to with better matchup
            for pokemon in battle.available_switches:
                if all(get_effectiveness(enemy_type, pokemon.types) <= 1.0 for enemy_type in battle.opponent_active_pokemon.types):
                    return pokemon.switch()
        
        #if no defensive switch, choose a move with best damage
        if battle.available_moves:
            return self.create_order(max(battle.available_moves, key=lambda move: move.base_power * get_effectiveness(move.type, battle.opponent_active_pokemon.types), default=None))
        #if no moves available, choose a random move
        return self.choose_random_move(battle) 
    
    def max_damage_move(self, battle: Battle):
        #function to find the move with maximum damage
        return max(
            battle.available_moves,
            key=lambda move: calculate_damage(
                move=move,
                attacker=battle.active_pokemon,
                defender=battle.opponent_active_pokemon,
                battle=battle
            )[0],
            default=None
        )
    
    def can_ko(self, battle: Battle):
        #function to check if we can KO the opponent with any move
        for move in battle.available_moves:
            possible_damage = calculate_damage(
                move=move,
                attacker=battle.active_pokemon,
                defender=battle.opponent_active_pokemon,
                battle=battle
            )
            # Check if the average damage is enough to KO the opponent
            if (possible_damage[0] + possible_damage[1]) / 2 >= battle.opponent_active_pokemon.current_hp:
                return True
        return False
    
    def speed(self, pokemon: Pokemon, battle: Battle):
        #calculates the speed of a certain pokemon
        #apply speed boosts
        boost = pokemon.boosts.get("spe", 0)
        speed = pokemon.stats["spe"]
        if boost > 0:
            speed *= (2 + boost) / 2
        elif boost < 0:
            speed *= 2 / (2 - boost)
        #apply paralysis
        if "par" in pokemon.status:
            speed /= 2
        if pokemon.item == "choicescarf":
            speed *= 1.5
        #doesnt account for abilities, terrain, weather, etc
        return speed
