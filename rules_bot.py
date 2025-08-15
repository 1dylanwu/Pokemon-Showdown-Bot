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

        #if we have bad type matchup, switch out
        #choose a pokemon with better matchup and higher hp
        if self.should_switch(battle):
            return self.create_order(self.defensive_switch(battle))
        
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
        if battle.available_moves:
            return self.create_order(self.max_damage_move(battle))
        
        #if our active pokemon is fainted, switch to a pokemon using same logic as defensive switch
        #this should be changed to be offensive(revenge kill)
        if(battle.active_pokemon is None and battle.available_switches):
            return self.create_order(self.defensive_switch(battle))
        
        return self.choose_random_move(battle)
    
    def should_switch(self, battle: Battle):
        #if we have bad defensive type matchup, switch out
        #doesnt account for alternatively being able to terrastalize
        for enemy_type in battle.opponent_active_pokemon.types:
            #checks enemy STAB typing against our active pokemon
            if battle.active_pokemon.damage_multiplier(enemy_type) > 1.0:
                return True
        return False

    def defensive_switch(self, battle: Battle):
            #find a pokemon to switch to with better matchup
            #this does not account for coverage moves
            best_score = -10
            best_switch = None
            for pokemon in battle.available_switches:
                #calculate score based on current hp and enemy type matchups
                #higher score means better switch
                score = pokemon.current_hp_fraction
                for enemy_type in battle.opponent_active_pokemon.types:
                    score -= pokemon.damage_multiplier(enemy_type)
                if score > best_score:
                    best_score = score
                    best_switch = pokemon
            return best_switch

                
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

    def can_set_hazard(self, move: Move, battle: Battle):
        #checks if a move is hazard setting and if we should use it
        #lists max number of stacks for each hazard
        HAZARD_MOVES = {
            "stealthrock":1,
            "toxicspikes":2,
            "spikes":3,
            "stickyweb":1
        }
        if move.side_condition and battle.side_conditions.get(move.side_condition, 0) < HAZARD_MOVES.get(move.side_condition, 0):
            #checks current side conditions to see if we should set it
            return True
        return False