from poke_env import Player
from poke_env.battle import Move, Pokemon, Battle
from poke_env.calc import calculate_damage, calculate_base_power
from type_chart import get_effectiveness

class RulesBot(Player):
    def choose_move(self, battle: Battle):
        
        #if we can KO the opponent and we outspeed, use the move that can KO
        #this does not account for priority moves
        if self.can_ko(battle) and self.speed(battle.active_pokemon, battle) >= self.speed(battle.opponent_active_pokemon, battle):
            return self.create_order(self.max_damage_move(battle))
        return self.choose_random_move(battle)
        '''
        #if we have bad defensive type matchup, switch out
        #doesnt account for alternatively being able to terrastalize
        defensive_switch = False
        for enemy_type in battle.opponent_active_pokemon.types:
            #checks enemy STAB typing against our active pokemon
            if get_effectiveness(enemy_type, battle.active_pokemon.types) > 1.0:
                defensive_switch = True

        if defensive_switch:
            #find a pokemon to switch to with better matchup
            #this does not account for coverage moves or pokemon bulk/current hp
            for pokemon in battle.available_switches:
                if all(get_effectiveness(enemy_type, pokemon.types) <= 1.0 for enemy_type in battle.opponent_active_pokemon.types):
                    return pokemon.switch()
        
        #if we have HP to spare, prioritize using available status/setup/hazard moves
        if(battle.active_pokemon.current_hp_fraction > 0.6):
            for move in battle.available_moves:
                #prioritize setting hazards, then status moves, then setup moves
                if self.can_set_hazard(move, battle):
                    return self.create_order(move)
                if move.status and not battle.opponent_active_pokemon.status:
                    return self.create_order(move)
                if move.self_boost:
                    return self.create_order(move)
        else:
            #if low hp check for healing moves
            for move in battle.available_moves:
                if move.heal:
                    return self.create_order(move)
                
        return self.choose_random_move(battle)
        '''
                


    def max_damage_move(self, battle: Battle):
        #function to find the move with maximum damage
        return max(
            battle.available_moves,
            key=lambda move: calculate_damage(
                battle.active_pokemon.identifier("p2"),
                battle.opponent_active_pokemon.identifier("p1"),
                move,
                battle,
                False
            )[0],
            default=None
        )
    
    def can_ko(self, battle: Battle):
        #function to check if we can KO the opponent with any move
        for move in battle.available_moves:
            possible_damage = calculate_damage(
                battle.active_pokemon.identifier("p2"),
                battle.opponent_active_pokemon.identifier("p1"),
                move,
                battle,
                False
            )
            #check if the average damage is enough to KO the opponent
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
        if pokemon.status == "par":
            speed /= 2
        if pokemon.item == "choicescarf":
            speed *= 1.5
        #doesnt account for abilities, terrain, weather, etc
        return speed

    def can_set_hazard(move: Move, battle: Battle):
        #checks if a move is hazard setting and if we should use it
        #lists max number of stacks for each hazard
        HAZARD_MOVES = {
            "stea:lthrock":1,
            "toxicspikes":2,
            "spikes":3,
            "stickyweb":1
        }
        if move.side_condition and battle.side_conditions.get(move.side_condition, 0) < HAZARD_MOVES.get(move.side_condition, 0):
            #checks current side conditions to see if we should set it
            return True
        return False