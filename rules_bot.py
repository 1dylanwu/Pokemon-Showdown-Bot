from poke_env import Player
from poke_env.battle import Move, Pokemon, Battle
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
    